
import numpy as np
import pickle
import glob
import pandas as pd
from collections import defaultdict

class cell_data(object):
    def __init__(self, contour_folder_path, graph_folder_path):
        self.contours = defaultdict(dict)
        self.contour_folder_path = contour_folder_path
        self.graph_folder_path = graph_folder_path
        self.load_graphs(graph_folder_path)
        try:
            self.load_contours(contour_folder_path)
        except: pass

    def load_contours(self, contour_folder_path):
        print('Loading contours')
        files = glob.glob(str(contour_folder_path) + '/*.pkl')
        for path in files:
            key = path.split('\\')[-1].split('.')[0]
            with open(path, 'rb') as f:
                contours_cyto, labels_cyto, contours_nuclei, labels_nuclei = pickle.load(f)
                self.contours[key]['Cell'] = dict(zip(labels_cyto, contours_cyto))
                self.contours[key]['Nuclei'] = dict(zip(labels_nuclei, contours_nuclei))

    def load_graphs(self, graph_folder_path):
        print('Loading graphs')
        files = glob.glob(str(graph_folder_path) + '/*.pkl')
        df1 = pd.DataFrame({'Path':files})
        df2 = pd.DataFrame([f.split('\\')[-1].split('.')[0].split('_') for f in files], columns=['Condition', 'FOV', 'Cell'])
        self.df = pd.concat([df2, df1], axis=1)