from copy import deepcopy
from numpy import array as np_array, asarray as np_asarray, rint as np_rint
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from PIL.Image import fromarray


names = [
    'example_idx',
    'subj_obj_idx_local',
    'subj_obj_category_idx',
    'subj_gtbox_1',
    'subj_gtbox_2',
    'subj_gtbox_3',
    'subj_gtbox_4',
    'obj_obj_idx_local',
    'obj_obj_category_idx',
    'obj_gtbox_1',
    'obj_gtbox_2',
    'obj_gtbox_3',
    'obj_gtbox_4',
    'rel_local_idx',
    'rel_category_idx',
]


class GraftAugmenterDataset(Dataset):
    def __init__(self, dataset):
        from maskrcnn_benchmark.data import VGStats
        dataset = deepcopy(dataset)
        self.transforms = dataset.transforms
        dataset.transforms = None
        self.get_img_info = dataset.get_img_info
        self.dataset = dataset
        print('GraftAugmenterDataset.__init__: started running get_dataset_statistics')
        statistics = dataset.get_statistics()
        print('GraftAugmenterDataset.__init__: finished running get_dataset_statistics')
        stats = statistics['stats']
        print('GraftAugmenterDataset.__init__: started creating VGStats singleton')
        vg_stats = VGStats(
            statistics['fg_matrix'],
            statistics['pred_dist'],
            statistics['obj_classes'],
            statistics['rel_classes'],
            statistics['att_classes'],
            stats,
        )
        print('GraftAugmenterDataset.__init__: finished creating VGStats singleton')
        del vg_stats
        print('GraftAugmenterDataset.__init__: started converting stats to df')
        df_stats = DataFrame(stats, columns=names)
        print('GraftAugmenterDataset.__init__: finished converting stats to df')
        del stats
        print('GraftAugmenterDataset.__init__: started converting stats to df')
        rels_bottom_k = set(df_stats.groupby(['rel_category_idx']).count()['example_idx'].nsmallest(30).index.tolist())
        self.df_stats = df_stats
        print('GraftAugmenterDataset.__init__: started querying for least frequent relations')
        self.df_stats_bottom_k = df_stats.query("rel_category_idx.isin(@rels_bottom_k).values")
        self.img_info = [dataset.img_info[i] for i in self.df_stats_bottom_k['example_idx']]
        print('GraftAugmenterDataset.__init__: finished querying for least frequent relations')

    def swap(self, idx_og, subj_or_obj):
        '''
        swap operates on each rare triplet. # sample size = num_iamges * num_reare_triplet_in_image
        '''
        dataset = self.dataset
        df_stats = self.df_stats
        df_stats_bottom_k = self.df_stats_bottom_k
        row_og = df_stats_bottom_k.loc[idx_og, :]
        example_idx_og = row_og['example_idx']
        img_og, target_og, _ = dataset[example_idx_og]
        img_og_np = np_array(img_og)

        if subj_or_obj == 'subj':
            obj_category_idx_og = row_og['subj_obj_category_idx']
            gtbox_1_og = row_og['subj_gtbox_1']
            gtbox_2_og = row_og['subj_gtbox_2']
            gtbox_3_og = row_og['subj_gtbox_3']
            gtbox_4_og = row_og['subj_gtbox_4']
        else:
            obj_category_idx_og = row_og['obj_obj_category_idx']
            gtbox_1_og = row_og['obj_gtbox_1']
            gtbox_2_og = row_og['obj_gtbox_2']
            gtbox_3_og = row_og['obj_gtbox_3']
            gtbox_4_og = row_og['obj_gtbox_4']

        rows = df_stats[(df_stats['subj_obj_category_idx'] == obj_category_idx_og) | (df_stats['obj_obj_category_idx'] == obj_category_idx_og)]
        row_new = rows.sample(n=1, ignore_index=True)
        row_new = row_new.loc[0, :]

        idx_new = row_new['example_idx']
        img_new, _, _ = dataset[idx_new]
        subj_or_obj_new = 'subj' if row_new['subj_obj_category_idx'] == obj_category_idx_og else 'obj'

        if subj_or_obj_new == 'subj':
            gtbox_1_new = row_new['subj_gtbox_1']
            gtbox_2_new = row_new['subj_gtbox_2']
            gtbox_3_new = row_new['subj_gtbox_3']
            gtbox_4_new = row_new['subj_gtbox_4']
        else:
            gtbox_1_new = row_new['obj_gtbox_1']
            gtbox_2_new = row_new['obj_gtbox_2']
            gtbox_3_new = row_new['obj_gtbox_3']
            gtbox_4_new = row_new['obj_gtbox_4']

        img_new_np = np_asarray(img_new)
        roi_new_np = img_new_np[gtbox_2_new:gtbox_4_new, gtbox_1_new:gtbox_3_new, :]
        roi_new = fromarray(roi_new_np)
        roi_new_resized = resize(roi_new, [gtbox_4_og-gtbox_2_og, gtbox_3_og-gtbox_1_og]) # takes H,W
        roi_new_resized_np = np_asarray(roi_new_resized)
        img_og_np[gtbox_2_og:gtbox_4_og, gtbox_1_og:gtbox_3_og, :] = np_rint(0.5 * img_og_np[gtbox_2_og:gtbox_4_og, gtbox_1_og:gtbox_3_og, :] + 0.5 * roi_new_resized_np)
        img_og_modified = fromarray(img_og_np)
        return img_og_modified, target_og, None # set index to None

    def __getitem__(self, index):
        idx_og = self.df_stats_bottom_k.index[index]
        img, target, index = self.swap(idx_og, 'subj')
        img, target = self.transforms(img, target)
        return img, target, idx_og

    def __len__(self):
        return len(self.df_stats_bottom_k)
