from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import pickle
from rdkit import Chem
from ctsynther.utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph
from ctsynther.utils.smiles_utils import canonical_smiles, canonical_smiles_with_am, remove_am_without_canonical, \
    extract_relative_mapping, get_nonreactive_mask, randomize_smiles_with_am

from functools import partial
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from ctsynther.dataset import RetroDataset
from ctsynther.models.model import RetroModel


def load_checkpoint(args, model):
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer = checkpoint['optim']
    step = checkpoint['step']
    step += 1
    return step, optimizer, model.to(args.device)


def build_model(args, vocab_itos_src, vocab_itos_tgt):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

    model = RetroModel(
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
        vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
        shared_vocab=args.shared_vocab == 'True', shared_encoder=args.shared_encoder == 'True',
        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)

    return model.to(args.device)


def build_iterator(args, train=True, sample=False, augment=False):
    if train:
        # dataset
        dataset = RetroDataset(mode='train', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class == 'True',
                               shared_vocab=args.shared_vocab == 'True', sample=sample, augment=augment)
        dataset_val = RetroDataset(mode='val', data_folder=args.data_dir,
                                   intermediate_folder=args.intermediate_dir,
                                   known_class=args.known_class == 'True',
                                   shared_vocab=args.shared_vocab == 'True', sample=sample)
        
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']

        # dataloader
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn_nag, shuffle=not sample,  # num_workers=8,
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device, num_pos=args.batch_size_trn_pos))

        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))

        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos

    else:
        dataset = RetroDataset(mode='test', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class == 'True',
                               shared_vocab=args.shared_vocab == 'True')
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
            collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device, mode='test'))
                               # collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset




def cont_parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False, intermediate_folder=None, 
    vocab_file=None, known_class=False, max_src_length = 0, max_tgt_length=0):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''

        with open(os.path.join(intermediate_folder, vocab_file), 'rb') as f:
                    src_itos, tgt_itos = pickle.load(f)
        src_stoi = {src_itos[i]: i for i in range(len(src_itos))}
        tgt_stoi = {tgt_itos[i]: i for i in range(len(tgt_itos))}


        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Get the smiles graph
        smiles_graph = SmilesGraph(cano_prod)
        # Get the nonreactive masking based on atom-mapping
        gt_nonreactive_mask = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1)
        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
        if known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token
        gt_nonreactive_mask = [True] + gt_nonreactive_mask

        src_token = [src_stoi.get(st, src_stoi['<unk>']) for st in src_token]
        tgt_token = [tgt_stoi.get(tt, tgt_stoi['<unk>']) for tt in tgt_token]
         
        
        return src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask



def collate_fn(data, src_pad, tgt_pad, device='cuda', mode="train", num_pos=3):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)

    if mode=='test' :
        src, src_graph, tgt, alignment, nonreactive_mask, my_file = zip(*data)
        max_src_length = max([len(s) for s in src])
        max_tgt_length = max([len(t) for t in tgt])

        anchor = torch.zeros([], device=device)

        # Graph structure with edge attributes
        new_bond_matrix = anchor.new_zeros((len(data), max_src_length, max_src_length, 7), dtype=torch.long)

        # Pad_sequence
        new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
        new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)
        new_alignment = anchor.new_zeros((len(data), max_tgt_length - 1, max_src_length), dtype=torch.float)
        new_nonreactive_mask = anchor.new_ones((max_src_length, len(data)), dtype=torch.bool)

        for i in range(len(data)):
            new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
            # new_nonreactive_mask[:, i][:len(nonreactive_mask[i])] = torch.BoolTensor(nonreactive_mask[i])
            new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
            new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

            full_adj_matrix = torch.from_numpy(src_graph[i].full_adjacency_tensor)
            new_bond_matrix[i, 1:full_adj_matrix.shape[0]+1, 1:full_adj_matrix.shape[1]+1] = full_adj_matrix
    
        return new_src, new_tgt, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph), False

    

    # this is train and val
    src, src_graph, tgt, alignment, nonreactive_mask, my_file = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    """important!"""
    batch_size_trn = len(src)
    known_class = False
    intermediate_dir = './intermediate'
    p_num = num_pos
    """important!"""

    # turple --> list
    src = list(src)
    src_graph = list(src_graph)
    tgt = list(tgt)
    alignment = list(alignment)
    nonreactive_mask = list(nonreactive_mask)

    # quit()
    index = random.randint(0, batch_size_trn-1)
    prod, react, rt = my_file[index]
    
    # print(src.shape, tgt.shape, gt_context_alignment.shape, gt_nonreactive_mask.shape, bond.shape)
    cont_label = [0 for _ in range (batch_size_trn)]
    cont_label[index] = 1
    
    add_src, add_src_graph, add_tgt, add_alignment, add_nonreactive_mask = src, src_graph, tgt, alignment, nonreactive_mask   
    for i in range(p_num):
        new_src, new_src_graph, new_tgt, new_alignment, new_nonreactive_mask = cont_parse_smi(prod, react, rt, randomize=True, 
            intermediate_folder=intermediate_dir, vocab_file = 'vocab_share.pk', known_class=known_class, max_src_length= max_src_length, max_tgt_length= max_tgt_length)
            # print(new_src)
        pos = random.randint(0, batch_size_trn-1)
        # print(pos)
        src.insert(pos, new_src)
        src_graph.insert(pos, new_src_graph)
        tgt.insert(pos, new_tgt)
        alignment.insert(pos, new_alignment)
        nonreactive_mask.insert(pos, new_nonreactive_mask)
        cont_label.insert(pos, 1)
     
    # list --> tuple
    src = tuple(src)
    src_graph = tuple(src_graph)
    tgt = tuple(tgt)
    alignment = tuple(alignment)
    nonreactive_mask = tuple(nonreactive_mask)
    
    cont_label = torch.tensor(cont_label) 
    # print(cont_label)

    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    anchor = torch.zeros([], device=device)

    # Graph structure with edge attributes
    new_bond_matrix = anchor.new_zeros((len(data)+p_num, max_src_length, max_src_length, 7), dtype=torch.long)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)+p_num), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)+p_num), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((len(data)+p_num, max_tgt_length - 1, max_src_length), dtype=torch.float)
    new_nonreactive_mask = anchor.new_ones((max_src_length, len(data)+p_num), dtype=torch.bool)

    for i in range(len(data)+p_num):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_nonreactive_mask[:, i][:len(nonreactive_mask[i])] = torch.BoolTensor(nonreactive_mask[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

        full_adj_matrix = torch.from_numpy(src_graph[i].full_adjacency_tensor)
        new_bond_matrix[i, 1:full_adj_matrix.shape[0]+1, 1:full_adj_matrix.shape[1]+1] = full_adj_matrix
    
    return new_src, new_tgt, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph), cont_label
