from config import TIMESTAMPS, SHAPES, ONTOLOGY_DIR
from plot import plot_every_slice
import numpy as np
import json



struct_graph = json.load(open('structure_graph.json'))['msg'][0]
parent_level = 2 # level of node that should be the parent

id_2_belong_id = {}
belong_id_2_name = {}

def dfs(node, depth, belong_id):
    if node == None:
        return 
    if depth <= parent_level:
        # print('new')
        # print(node)
        belong_id = node['id']
        belong_id_2_name[node['id']] = node['name']
    
    id_2_belong_id[node['id']] = belong_id
    # print('id:', node['id'])
    for child in node['children']:
        dfs(child, depth + 1, belong_id)

dfs(struct_graph, 0, 0)





belong_id_2_region_id = {}
belong_id_2_region_id[0] = 0
prev_part = 0
region_id = 1

regions = [] # (struct id, region name)

for timestamp, shape in zip(TIMESTAMPS, SHAPES):
    grid = np.fromfile(f'{ONTOLOGY_DIR}{timestamp}/gridAnnotation.raw', dtype=np.int32, count=-1, sep='')

    struct = np.zeros_like(grid, dtype=np.int32)

    for i in range(len(grid)):
        id = grid[i]
        if id == 0:
            continue
        if id not in id_2_belong_id:
            print('empty id', id)
            continue
        
        belong_id = id_2_belong_id[id]
        if  belong_id not in belong_id_2_region_id:
            belong_id_2_region_id[belong_id] = region_id
            regions.append((region_id, belong_id_2_name[belong_id]))
            region_id += 1
        
        struct[i] = belong_id_2_region_id[belong_id]
        # print(grid[i])
        
    struct = np.reshape(struct, shape)
    

    # print(timestamp)
    # print('unique',np.unique(struct))


    # npart = len(np.unique(grid))
    # print(timestamp, "new:", len(belong2value) - prev_part, "old:", npart - (len(belong2value) - prev_part))
    # prev_part = len(belong2value)
    

    plot_every_slice(struct, save_path=f'ontology_{timestamp}.png')
    np.save(f'ontology_{timestamp}.npy', struct)
    

print(regions)
