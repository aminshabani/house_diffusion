import math
import random
import torch as th

from PIL import Image, ImageDraw
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
import os
import cv2 as cv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
import copy

def load_rplanhg_data(
    batch_size,
    analog_bit,
    target_set = 8,
    set_name = 'train',
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of target set {target_set}")
    deterministic = False if set_name=='train' else True
    dataset = RPlanhgDataset(set_name, analog_bit, target_set)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader

def make_non_manhattan(poly, polygon, house_poly):
    dist = abs(poly[2]-poly[0])
    direction = np.argmin(dist)
    center = poly.mean(0)
    min = poly.min(0)
    max = poly.max(0)

    tmp = np.random.randint(3, 7)
    new_min_y = center[1]-(max[1]-min[1])/tmp
    new_max_y = center[1]+(max[1]-min[1])/tmp
    if center[0]<128:
        new_min_x = min[0]-(max[0]-min[0])/np.random.randint(2,5)
        new_max_x = center[0]
        poly1=[[min[0], min[1]], [new_min_x, new_min_y], [new_min_x, new_max_y], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]]]
    else:
        new_min_x = center[0]
        new_max_x = max[0]+(max[0]-min[0])/np.random.randint(2,5)
        poly1=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [new_max_x, new_max_y], [new_max_x, new_min_y], [max[0], min[1]]]

    new_min_x = center[0]-(max[0]-min[0])/tmp
    new_max_x = center[0]+(max[0]-min[0])/tmp
    if center[1]<128:
        new_min_y = min[1]-(max[1]-min[1])/np.random.randint(2,5)
        new_max_y = center[1]
        poly2=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]], [new_max_x, new_min_y], [new_min_x, new_min_y]]
    else:
        new_min_y = center[1]
        new_max_y = max[1]+(max[1]-min[1])/np.random.randint(2,5)
        poly2=[[min[0], min[1]], [min[0], max[1]], [new_min_x, new_max_y], [new_max_x, new_max_y], [max[0], max[1]], [max[0], min[1]]]
    p1 = gm.Polygon(poly1)
    iou1 = house_poly.intersection(p1).area/ p1.area
    p2 = gm.Polygon(poly2)
    iou2 = house_poly.intersection(p2).area/ p2.area
    if iou1>0.9 and iou2>0.9:
        return poly
    if iou1<iou2:
        return poly1
    else:
        return poly2

get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]
class RPlanhgDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False):
        super().__init__()
        base_dir = '../datasets/rplan'
        self.non_manhattan = non_manhattan
        self.set_name = set_name
        self.analog_bit = analog_bit
        self.target_set = target_set
        self.subgraphs = []
        self.org_graphs = []
        self.org_houses = []
        max_num_points = 100
        if self.set_name == 'eval':
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
        if os.path.exists(f'processed_rplan/rplan_{set_name}_{target_set}.npz'):
            data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}.npz', allow_pickle=True)
            self.graphs = data['graphs']
            self.houses = data['houses']
            self.door_masks = data['door_masks']
            self.self_masks = data['self_masks']
            self.gen_masks = data['gen_masks']
            self.num_coords = 2
            self.max_num_points = max_num_points
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
            if self.set_name == 'eval':
                data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}_syn.npz', allow_pickle=True)
                self.syn_graphs = data['graphs']
                self.syn_houses = data['houses']
                self.syn_door_masks = data['door_masks']
                self.syn_self_masks = data['self_masks']
                self.syn_gen_masks = data['gen_masks']
        else:
            with open(f'{base_dir}/list.txt') as f:
                lines = f.readlines()
            cnt=0
            for line in tqdm(lines):
                cnt=cnt+1
                file_name = f'{base_dir}/{line[:-1]}'
                rms_type, fp_eds,rms_bbs,eds_to_rms=reader(file_name) 
                fp_size = len([x for x in rms_type if x != 15 and x != 17])
                if self.set_name=='train' and fp_size == target_set:
                        continue
                if self.set_name=='eval' and fp_size != target_set:
                        continue
                a = [rms_type, rms_bbs, fp_eds, eds_to_rms]
                self.subgraphs.append(a)
            for graph in tqdm(self.subgraphs):
                rms_type = graph[0]
                rms_bbs = graph[1]
                fp_eds = graph[2]
                eds_to_rms= graph[3]
                rms_bbs = np.array(rms_bbs)
                fp_eds = np.array(fp_eds)

                # extract boundary box and centralize
                tl = np.min(rms_bbs[:, :2], 0)
                br = np.max(rms_bbs[:, 2:], 0)
                shift = (tl+br)/2.0 - 0.5
                rms_bbs[:, :2] -= shift
                rms_bbs[:, 2:] -= shift
                fp_eds[:, :2] -= shift
                fp_eds[:, 2:] -= shift
                tl -= shift
                br -= shift

                # build input graph
                graph_nodes, graph_edges, rooms_mks = self.build_graph(rms_type, fp_eds, eds_to_rms)

                house = []
                for room_mask, room_type in zip(rooms_mks, graph_nodes):
                    room_mask = room_mask.astype(np.uint8)
                    room_mask = cv.resize(room_mask, (256, 256), interpolation = cv.INTER_AREA)
                    contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    contours = contours[0]
                    house.append([contours[:,0,:], room_type])
                self.org_graphs.append(graph_edges)
                self.org_houses.append(house)
            houses = []
            door_masks = []
            self_masks = []
            gen_masks = []
            graphs = []
            if self.set_name=='train':
                cnumber_dist = defaultdict(list)

            if self.non_manhattan:
                for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    # Generating non-manhattan Balconies
                    tmp = []
                    for i, room in enumerate(h):
                        if room[1]>10:
                            continue
                        if len(room[0])!=4: 
                            continue
                        if np.random.randint(2):
                            continue
                        poly = gm.Polygon(room[0])
                        house_polygon = unary_union([gm.Polygon(room[0]) for room in h])
                        room[0] = make_non_manhattan(room[0], poly, house_polygon)

            for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                house = []
                corner_bounds = []
                num_points = 0
                for i, room in enumerate(h):
                    if room[1]>10:
                        room[1] = {15:11, 17:12, 16:13}[room[1]]
                    room[0] = np.reshape(room[0], [len(room[0]), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                    room[0] = room[0] * 2 # map to [-1, 1]
                    if self.set_name=='train':
                        cnumber_dist[room[1]].append(len(room[0]))
                    # Adding conditions
                    num_room_corners = len(room[0])
                    rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                    room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                    corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                    # Src_key_padding_mask
                    padding_mask = np.repeat(1, num_room_corners)
                    padding_mask = np.expand_dims(padding_mask, 1)
                    # Generating corner bounds for attention masks
                    connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                    connections += num_points
                    corner_bounds.append([num_points, num_points+num_room_corners])
                    num_points += num_room_corners
                    room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
                    house.append(room)

                house_layouts = np.concatenate(house, 0)
                if len(house_layouts)>max_num_points:
                    continue
                padding = np.zeros((max_num_points-len(house_layouts), 94))
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                house_layouts = np.concatenate((house_layouts, padding), 0)

                door_mask = np.ones((max_num_points, max_num_points))
                self_mask = np.ones((max_num_points, max_num_points))
                for i in range(len(corner_bounds)):
                    for j in range(len(corner_bounds)):
                        if i==j:
                            self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                        elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                houses.append(house_layouts)
                door_masks.append(door_mask)
                self_masks.append(self_mask)
                gen_masks.append(gen_mask)
                graphs.append(graph)
            self.max_num_points = max_num_points
            self.houses = houses
            self.door_masks = door_masks
            self.self_masks = self_masks
            self.gen_masks = gen_masks
            self.num_coords = 2
            self.graphs = graphs

            np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}', graphs=self.graphs, houses=self.houses,
                    door_masks=self.door_masks, self_masks=self.self_masks, gen_masks=self.gen_masks)
            if self.set_name=='train':
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_cndist', cnumber_dist=cnumber_dist)

            if set_name=='eval':
                houses = []
                graphs = []
                door_masks = []
                self_masks = []
                gen_masks = []
                len_house_layouts = 0
                for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    house = []
                    corner_bounds = []
                    num_points = 0
                    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    while np.sum(num_room_corners_total)>=max_num_points:
                        num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    for i, room in enumerate(h):
                        # Adding conditions
                        num_room_corners = num_room_corners_total[i]
                        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                        room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                        # Src_key_padding_mask
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections), 1)
                        house.append(room)

                    house_layouts = np.concatenate(house, 0)
                    if np.sum([len(room[0]) for room in h])>max_num_points:
                        continue
                    padding = np.zeros((max_num_points-len(house_layouts), 94))
                    gen_mask = np.ones((max_num_points, max_num_points))
                    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                    house_layouts = np.concatenate((house_layouts, padding), 0)

                    door_mask = np.ones((max_num_points, max_num_points))
                    self_mask = np.ones((max_num_points, max_num_points))
                    for i, room in enumerate(h):
                        if room[1]==1:
                            living_room_index = i
                            break
                    for i in range(len(corner_bounds)):
                        is_connected = False
                        for j in range(len(corner_bounds)):
                            if i==j:
                                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                            elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                                door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                                is_connected = True
                        if not is_connected:
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

                    houses.append(house_layouts)
                    door_masks.append(door_mask)
                    self_masks.append(self_mask)
                    gen_masks.append(gen_mask)
                    graphs.append(graph)
                self.syn_houses = houses
                self.syn_door_masks = door_masks
                self.syn_self_masks = self_masks
                self.syn_gen_masks = gen_masks
                self.syn_graphs = graphs
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_syn', graphs=self.syn_graphs, houses=self.syn_houses,
                        door_masks=self.syn_door_masks, self_masks=self.syn_self_masks, gen_masks=self.syn_gen_masks)

    def __len__(self):
        return len(self.houses)

    def __getitem__(self, idx):
        # idx = int(idx//20)
        arr = self.houses[idx][:, :self.num_coords]
        graph = np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)

        cond = {
                'door_mask': self.door_masks[idx],
                'self_mask': self.self_masks[idx],
                'gen_mask': self.gen_masks[idx],
                'room_types': self.houses[idx][:, self.num_coords:self.num_coords+25],
                'corner_indices': self.houses[idx][:, self.num_coords+25:self.num_coords+57],
                'room_indices': self.houses[idx][:, self.num_coords+57:self.num_coords+89],
                'src_key_padding_mask': 1-self.houses[idx][:, self.num_coords+89],
                'connections': self.houses[idx][:, self.num_coords+90:self.num_coords+92],
                'graph': graph,
                }
        if self.set_name == 'eval':
            syn_graph = np.concatenate((self.syn_graphs[idx], np.zeros([200-len(self.syn_graphs[idx]), 3])), 0)
            assert (graph == syn_graph).all(), idx
            cond.update({
                'syn_door_mask': self.syn_door_masks[idx],
                'syn_self_mask': self.syn_self_masks[idx],
                'syn_gen_mask': self.syn_gen_masks[idx],
                'syn_room_types': self.syn_houses[idx][:, self.num_coords:self.num_coords+25],
                'syn_corner_indices': self.syn_houses[idx][:, self.num_coords+25:self.num_coords+57],
                'syn_room_indices': self.syn_houses[idx][:, self.num_coords+57:self.num_coords+89],
                'syn_src_key_padding_mask': 1-self.syn_houses[idx][:, self.num_coords+89],
                'syn_connections': self.syn_houses[idx][:, self.num_coords+90:self.num_coords+92],
                'syn_graph': syn_graph,
                })
        if self.set_name == 'train':
            #### Random Rotate
            rotation = random.randint(0,3)
            if rotation == 1:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 0] = -arr[:, 0]
            elif rotation == 2:
                arr[:, [0, 1]] = -arr[:, [1, 0]]
            elif rotation == 3:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 1] = -arr[:, 1]

            ## To generate any rotation uncomment this

            # if self.non_manhattan:
                # theta = random.random()*np.pi/2
                # rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                             # [np.sin(theta), np.cos(theta), 0]])
                # arr = np.matmul(arr,rot_mat)[:,:2]

            # Random Scale
            # arr = arr * np.random.normal(1., .5)

            # Random Shift
            # arr[:, 0] = arr[:, 0] + np.random.normal(0., .1)
            # arr[:, 1] = arr[:, 1] + np.random.normal(0., .1)

        if not self.analog_bit:
            arr = np.transpose(arr, [1, 0])
            return arr.astype(float), cond
        else:
            ONE_HOT_RES = 256
            arr_onehot = np.zeros((ONE_HOT_RES*2, arr.shape[1])) - 1
            xs = ((arr[:, 0]+1)*(ONE_HOT_RES/2)).astype(int)
            ys = ((arr[:, 1]+1)*(ONE_HOT_RES/2)).astype(int)
            xs = np.array([get_bin(x, 8) for x in xs])
            ys = np.array([get_bin(x, 8) for x in ys])
            arr_onehot = np.concatenate([xs, ys], 1)
            arr_onehot = np.transpose(arr_onehot, [1, 0])
            arr_onehot[arr_onehot==0] = -1
            return arr_onehot.astype(float), cond

    def make_sequence(self, edges):
        polys = []
        v_curr = tuple(edges[0][:2])
        e_ind_curr = 0
        e_visited = [0]
        seq_tracker = [v_curr]
        find_next = False
        while len(e_visited) < len(edges):
            if find_next == False:
                if v_curr == tuple(edges[e_ind_curr][2:]):
                    v_curr = tuple(edges[e_ind_curr][:2])
                else:
                    v_curr = tuple(edges[e_ind_curr][2:])
                find_next = not find_next 
            else:
                # look for next edge
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        if (v_curr == tuple(e[:2])):
                            v_curr = tuple(e[2:])
                            e_ind_curr = k
                            e_visited.append(k)
                            break
                        elif (v_curr == tuple(e[2:])):
                            v_curr = tuple(e[:2])
                            e_ind_curr = k
                            e_visited.append(k)
                            break

            # extract next sequence
            if v_curr == seq_tracker[-1]:
                polys.append(seq_tracker)
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        v_curr = tuple(edges[0][:2])
                        seq_tracker = [v_curr]
                        find_next = False
                        e_ind_curr = k
                        e_visited.append(k)
                        break
            else:
                seq_tracker.append(v_curr)
        polys.append(seq_tracker)

        return polys

    def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):
        # create edges
        triples = []
        nodes = rms_type 
        # encode connections
        for k in range(len(nodes)):
            for l in range(len(nodes)):
                if l > k:
                    is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                    if is_adjacent:
                        if 'train' in self.set_name:
                            triples.append([k, 1, l])
                        else:
                            triples.append([k, 1, l])
                    else:
                        if 'train' in self.set_name:
                            triples.append([k, -1, l])
                        else:
                            triples.append([k, -1, l])
        # get rooms masks
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):                  
            eds_to_rms_tmp.append([eds_to_rms[l][0]])
        rms_masks = []
        im_size = 256
        fp_mk = np.zeros((out_size, out_size))
        for k in range(len(nodes)):
            # add rooms and doors
            eds = []
            for l, e_map in enumerate(eds_to_rms_tmp):
                if (k in e_map):
                    eds.append(l)
            # draw rooms
            rm_im = Image.new('L', (im_size, im_size))
            dr = ImageDraw.Draw(rm_im)
            for eds_poly in [eds]:
                poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
                poly = [(im_size*x, im_size*y) for x, y in poly]
                if len(poly) >= 2:
                    dr.polygon(poly, fill='white')
                else:
                    print("Empty room")
                    exit(0)
            rm_im = rm_im.resize((out_size, out_size))
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr>0)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)
            if rms_type[k] != 15 and rms_type[k] != 17:
                fp_mk[inds] = k+1
        # trick to remove overlap
        for k in range(len(nodes)):
            if rms_type[k] != 15 and rms_type[k] != 17:
                rm_arr = np.zeros((out_size, out_size))
                inds = np.where(fp_mk==k+1)
                rm_arr[inds] = 1.0
                rms_masks[k] = rm_arr
        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        rms_masks = np.array(rms_masks)
        return nodes, triples, rms_masks

def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b
    h1, h2 = x1-x0, x3-x2
    w1, w2 = y1-y0, y3-y2
    xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
    yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0
    delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
    delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0
    delta = max(delta_x, delta_y)
    return delta < threshold

def reader(filename):
    with open(filename) as f:
        info =json.load(f)
        rms_bbs=np.asarray(info['boxes'])
        fp_eds=info['edges']
        rms_type=info['room_type']
        eds_to_rms=info['ed_rm']
        s_r=0
        for rmk in range(len(rms_type)):
            if(rms_type[rmk]!=17):
                s_r=s_r+1   
        rms_bbs = np.array(rms_bbs)/256.0
        fp_eds = np.array(fp_eds)/256.0 
        fp_eds = fp_eds[:, :4]
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl+br)/2.0 - 0.5
        rms_bbs[:, :2] -= shift 
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift 
        tl -= shift
        br -= shift
        return rms_type,fp_eds,rms_bbs,eds_to_rms

if __name__ == '__main__':
    dataset = RPlanhgDataset('eval', False, 8)
