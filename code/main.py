import os
import pickle
import gzip
import argparse
import time
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as PCA_scikit
from sklearn.manifold import TSNE, Isomap
from sklearn.manifold import MDS as MDS_scikit
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from manifold import MDS, ISOMAP
from pca import PCA, AlternativePCA, RobustPCA
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "animals":
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape
        print("n =", n)
        print("d =", d)

        f1, f2 = np.random.choice(d, size=2, replace=False)

        plt.figure()
        plt.scatter(X[:,f1], X[:,f2])
        plt.xlabel("$x_{%d}$" % f1)
        plt.ylabel("$x_{%d}$" % f2)
        for i in range(n):
            plt.annotate(animals[i], (X[i,f1], X[i,f2]))
        
        utils.savefig('two_random_features_animals.png','animals')

        #############################################################################
        ################# Principal components analysis #############################
        #############################################################################
        print('=============== PCA (SVD) ===============')
        n_pcs = 2
        model_pca = PCA(k = n_pcs)
        model_pca.fit(X)
        Z_pca = model_pca.compress(X)

        plt.figure()
        plt.scatter(Z_pca[:,0], Z_pca[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        for i in range(n):
            plt.annotate(animals[i], (Z_pca[i,0], Z_pca[i,1]))
        
        utils.savefig('PCA_animals.png', 'animals')

        model_pca_scikit = PCA_scikit(n_components = n_pcs)
        model_pca_scikit.fit(X)
        print(f'Variance explained (scikit): \n{model_pca_scikit.explained_variance_ratio_}, \n \
            ({np.sum(model_pca_scikit.explained_variance_ratio_)} total)\n')
        Z_pca_scikit = model_pca_scikit.transform(X)

        plt.figure()
        plt.scatter(Z_pca_scikit[:,0], Z_pca_scikit[:,1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        for i in range(n):
            plt.annotate(animals[i], (Z_pca_scikit[i,0], Z_pca_scikit[i,1]))
        
        utils.savefig('PCA_scikit_animals.png', 'animals')

        #############################################################################
        ##################### Multi-Dimensional Scaling #############################
        #############################################################################
        print('=============== MDS ===============')
        model_mds = MDS(n_components=2)
        Z_mds = model_mds.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z_mds[:,0], Z_mds[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS')
        for i in range(n):
            ax.annotate(animals[i], (Z_mds[i,0], Z_mds[i,1]))
        utils.savefig('MDS_animals.png', 'animals')

        model_mds_scikit = MDS_scikit(n_components=2)
        Z_mds_scikit = model_mds_scikit.fit_transform(X)

        fig, ax = plt.subplots()
        ax.scatter(Z_mds_scikit[:,0], Z_mds_scikit[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS scikit')
        for i in range(n):
            ax.annotate(animals[i], (Z_mds_scikit[i,0], Z_mds_scikit[i,1]))
        utils.savefig('MDS_scikit_animals.png', 'animals')

        #############################################################################
        ############################# ISOMAP ########################################
        #############################################################################
        print('=============== ISOMAP ===============')
        for n_neigh in [2,3]:
            model_isomap = ISOMAP(n_components=2, n_neighbours=n_neigh)
            Z_isomap = model_isomap.compress(X)

            fig, ax = plt.subplots()
            ax.scatter(Z_isomap[:,0], Z_isomap[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neigh)
            for i in range(n):
                ax.annotate(animals[i], (Z_isomap[i,0], Z_isomap[i,1]))
            utils.savefig('ISOMAP_nn%d_animals.png' % n_neigh, 'animals')

        for n_neigh in [2,3]:
            model_isomap_scikit = Isomap(n_neighbors=n_neigh, n_components=2)
            Z_isomap_scikit = model_isomap_scikit.fit_transform(X)

            fig, ax = plt.subplots()
            ax.scatter(Z_isomap_scikit[:,0], Z_isomap_scikit[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neigh)
            for i in range(n):
                ax.annotate(animals[i], (Z_isomap_scikit[i,0], Z_isomap_scikit[i,1]))
            utils.savefig('ISOMAP_scikit_nn%d_animals.png' % n_neigh, 'animals')

        #############################################################################
        ############################# t-SNE #########################################
        #############################################################################
        print('=============== t-SNE ===============')
        model_tsne = TSNE(n_components=2)
        Z_tsne = model_tsne.fit_transform(X)

        fig, ax = plt.subplots()
        ax.scatter(Z_tsne[:,0], Z_tsne[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('t-SNE')
        for i in range(n):
            ax.annotate(animals[i], (Z_tsne[i,0], Z_tsne[i,1]))
        utils.savefig('tSNE_animals.png', 'animals')
            
    elif question == 'hcv':
        filename = "hcvdat0.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            hcv = pd.read_csv(f)
        hcv = hcv.drop(hcv.columns[[0]], axis=1).dropna()
        hcv = hcv[hcv["Category"] != "0s=suspect Blood Donor"]

        # Changing Sex = {"m", "f"} to Sex = {0, 1}
        # print(np.unique(X["Sex"]))
        hcv["Sex"] = hcv["Sex"].replace(to_replace = {"m" : 0, "f" : 1})

        # Drop the "Category" (label) column and convert to numpy array
        X = hcv.drop(hcv.columns[[0, 1, 2]], axis=1).to_numpy()
        y = hcv["Category"]
        print(y)
        # print(X)
        print(f'X shape: {X.shape}')
        print(f'y.shape: {y.shape}')

        #############################################################################
        ################# Principal components analysis #############################
        #############################################################################
        print('=============== PCA (SVD) ===============')
        n_pcs = 2
        model_pca = PCA(k = n_pcs)
        model_pca.fit(X)
        Z_pca = model_pca.compress(X)

        # print(Z_pca.shape)
        # Z_pca = np.append(Z_pca, y.reshape((y.size, 1)), axis=1)
        # print(Z_pca.shape)

        y_num = np.unique(y, return_inverse=True)[1].tolist()
        # y_colors = {"0=Blood Donor":'red', "1=Hepatitis":'green', "2=Fibrosis":'blue', "3=Cirrhosis":'black'}

        plt.figure()
        # plt.scatter(Z_pca[:,0], Z_pca[:,1], alpha=0.5,c=y.map(y_colors))
        plt.scatter(Z_pca[:,0], Z_pca[:,1], alpha=0.5,c=y_num)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title('PCA (SVD)')
        utils.savefig('PCA_hcv.png', 'HCV')


        model_pca_scikit = PCA_scikit(n_components = n_pcs)
        model_pca_scikit.fit(X)
        print(f'Variance explained (scikit): \n{model_pca_scikit.explained_variance_ratio_}, \n \
            ({np.sum(model_pca_scikit.explained_variance_ratio_)} total)\n')
        Z_pca_scikit = model_pca_scikit.transform(X)

        plt.figure()
        plt.scatter(Z_pca_scikit[:,0], Z_pca_scikit[:,1], alpha=0.5,c=y_num)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title('PCA (scikit)')
        utils.savefig('PCA_scikit_hcv.png', 'HCV')

        #############################################################################
        ##################### Multi-Dimensional Scaling #############################
        #############################################################################
        print('=============== MDS ===============')
        # model_mds = MDS(n_components=2)
        # Z_mds = model_mds.compress(X)

        # fig, ax = plt.subplots()
        # ax.scatter(Z_mds[:,0], Z_mds[:,1])
        # plt.ylabel('z2')
        # plt.xlabel('z1')
        # plt.title('MDS')
        # utils.savefig('MDS_hcv.png', 'HCV')
        y_num = np.unique(y, return_inverse=True)[1].tolist()

        model_mds_scikit = MDS_scikit(n_components=2)
        Z_mds_scikit = model_mds_scikit.fit_transform(X)

        fig, ax = plt.subplots()
        ax.scatter(Z_mds_scikit[:,0], Z_mds_scikit[:,1], alpha=0.5,c=y_num)
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS (scikit)')
        utils.savefig('MDS_scikit_hcv.png', 'HCV')

        #############################################################################
        ############################# ISOMAP ########################################
        #############################################################################
        print('=============== ISOMAP ===============')
        # for n_neigh in [2,3]:
        #     model_isomap = ISOMAP(n_components=2, n_neighbours=n_neigh)
        #     Z_isomap = model_isomap.compress(X)

        #     fig, ax = plt.subplots()
        #     ax.scatter(Z_isomap[:,0], Z_isomap[:,1])
        #     plt.ylabel('z2')
        #     plt.xlabel('z1')
        #     plt.title('ISOMAP with NN=%d' % n_neigh)
        #     utils.savefig('ISOMAP_nn%d_hcv.png' % n_neigh, 'HCV')
        y_num = np.unique(y, return_inverse=True)[1].tolist()

        for n_neigh in [2,3]:
            model_isomap_scikit = Isomap(n_neighbors=n_neigh, n_components=2)
            Z_isomap_scikit = model_isomap_scikit.fit_transform(X)

            fig, ax = plt.subplots()
            ax.scatter(Z_isomap_scikit[:,0], Z_isomap_scikit[:,1], alpha=0.5,c=y_num)
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neigh)
            utils.savefig('ISOMAP_scikit_nn%d_hcv.png' % n_neigh, 'HCV')

        #############################################################################
        ############################# t-SNE #########################################
        #############################################################################
        print('=============== t-SNE ===============')
        model_tsne = TSNE(n_components=2)
        Z_tsne = model_tsne.fit_transform(X)

        y_num = np.unique(y, return_inverse=True)[1].tolist()

        fig, ax = plt.subplots()
        ax.scatter(Z_tsne[:,0], Z_tsne[:,1], alpha=0.5,c=y_num)
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('t-SNE')
        utils.savefig('tSNE_hcv.png', 'HCV')
    
    elif question == 'RNAseq':
        # filename = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"
        # with tarfile.open(os.path.join("..", "data", filename)) as tar:
            # tar.extractall()
            # tar.close()
        filename1 = "TCGA-PANCAN-HiSeq-801x20531.csv"
        filename2 = "TCGA-PANCAN-HiSeq-801x20531_labels.csv"
        with open(os.path.join("..", "data", "TCGA-PANCAN-HiSeq-801x20531", filename1), "rb") as f:
            X_data = pd.read_csv(f)
        with open(os.path.join("..", "data", "TCGA-PANCAN-HiSeq-801x20531", filename2), "rb") as f:
            y_data = pd.read_csv(f)
       
        X = X_data.drop(X_data.columns[[0]], axis=1)
        y = y_data.drop(y_data.columns[[0]], axis=1)

        print("========== PCA ==========")

        n_pcs = 2
        model_pca = PCA(k = n_pcs)
        model_pca.fit(X)
        Z_pca = model_pca.compress(X)
        df_pca = pd.DataFrame(Z_pca.to_numpy(), columns=['z1', 'z2'])
        df_pca['label'] = y.to_numpy()
        groups_pca = df_pca.groupby('label')

        fig, ax = plt.subplots()
        for name, group in groups_pca:
            ax.scatter(group.z1, group.z2, alpha=0.5, label=name)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title('Principal Components Analysis')
        ax.legend()
    
        utils.savefig('PCA_RNAseq.png', 'cancerGeneExpressionRNASeq')



        n_pcs = 2
        model_pca_scikit = PCA_scikit(n_components = n_pcs)
        model_pca_scikit.fit(X)
        print(f'Variance explained (scikit): \n{model_pca_scikit.explained_variance_ratio_}, \n \
            ({np.sum(model_pca_scikit.explained_variance_ratio_)} total)\n')

        Z_pca_scikit = model_pca_scikit.transform(X)
        
        df_pca_scikit = pd.DataFrame(Z_pca_scikit, columns=['z1', 'z2'])
        df_pca_scikit['label'] = y.to_numpy()
        
        groups_pca_scikit = df_pca_scikit.groupby('label')

        fig, ax = plt.subplots()
        for name, group in groups_pca_scikit:
            ax.scatter(group.z1, group.z2, alpha=0.5, label=name)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title('Principal Components Analysis (scikit)')
        ax.legend()
    
        utils.savefig('PCA_scikit_RNAseq.png', 'cancerGeneExpressionRNASeq')

        print("========== MDS ==========")
        # model_mds = MDS(n_components=2)
        # Z_mds = model_mds.compress(X)

        # df_mds = pd.DataFrame(Z_mds, columns=['z1', 'z2'])
        # df_mds['label'] = y.to_numpy()
        
        # groups_mds = df_mds.groupby('label')

        # fig, ax = plt.subplots()
        # for name, group in groups_mds:
        #     ax.scatter(group.z1, group.z2, alpha=0.5, label=name)
        # plt.ylabel('Z2')
        # plt.xlabel('Z1')
        # plt.title('Multi-Dimensional Scaling')
        # ax.legend()
    
        # utils.savefig('MDS_RNAseq.png', 'cancerGeneExpressionRNASeq')

        model_mds_scikit = MDS_scikit(n_components=2)       
        Z_mds_scikit = model_mds_scikit.fit_transform(X)

        df_mds_scikit = pd.DataFrame(Z_mds_scikit, columns=['z1', 'z2'])
        df_mds_scikit['label'] = y.to_numpy()
        
        groups_mds_scikit = df_mds_scikit.groupby('label')

        fig, ax = plt.subplots()
        for name, group in groups_mds_scikit:
            ax.scatter(group.z1, group.z2, alpha=0.5, label=name)
        plt.ylabel('Z2')
        plt.xlabel('Z1')
        plt.title('Multi-Dimensional Scaling (scikit)')
        ax.legend()
    
        utils.savefig('MDS_scikit_RNAseq.png', 'cancerGeneExpressionRNASeq')

        # fig, ax = plt.subplots()
        # ax.scatter(Z_mds_scikit[:,0], Z_mds_scikit[:,1], alpha=0.5,)
        # plt.ylabel('z2')
        # plt.xlabel('z1')
        # plt.title('MDS (scikit)')
        # utils.savefig('MDS_scikit_RNAseq.png', 'cancerGeneExpressionRNASeq')

        # print("========== ISOMAP ==========")
        for n_neigh in [1, 3, 10, 30]:
            model_isomap_scikit = Isomap(n_neighbors=n_neigh, n_components=2)
            Z_isomap_scikit = model_isomap_scikit.fit_transform(X)
            
            df_isomap = pd.DataFrame(Z_isomap_scikit, columns=['z1', 'z2'])
            df_isomap['label'] = y.to_numpy()
            
            groups_isomap = df_isomap.groupby('label')

            fig, ax = plt.subplots()
            for name, group in groups_isomap:
                ax.scatter(group.z1, group.z2, alpha=0.5, label=name)
            plt.ylabel('Z2')
            plt.xlabel('Z1')
            ax.legend()
            plt.title('ISOMAP (with NN=%d)' % n_neigh)
            utils.savefig('ISOMAP_scikit_nn%d_RNAseq.png' % n_neigh, 'cancerGeneExpressionRNASeq')

        #     fig, ax = plt.subplots()
        #     ax.scatter(Z_isomap_scikit[:,0], Z_isomap_scikit[:,1], alpha=0.5)
        #     plt.ylabel('z2')
        #     plt.xlabel('z1')
        #     plt.title('ISOMAP with NN=%d' % n_neigh)
        #     utils.savefig('ISOMAP_scikit_nn%d_RNAseq.png' % n_neigh, 'cancerGeneExpressionRNASeq')

        # print("========== t-SNE ==========")
        model_tsne = TSNE(n_components=2)
        Z_tsne = model_tsne.fit_transform(X)

        df_tsne = pd.DataFrame(Z_tsne, columns=['z1', 'z2'])
        df_tsne['label'] = y.to_numpy()
        
        groups_tsne = df_tsne.groupby('label')

        fig, ax = plt.subplots()
        for name, group in groups_tsne:
            ax.scatter(group.z1, group.z2, alpha=0.5, label=name)
        plt.ylabel('Z2')
        plt.xlabel('Z1')
        plt.title('t-SNE')
        ax.legend()
    
        utils.savefig('tSNE_RNAseq.png', 'cancerGeneExpressionRNASeq')

        # fig, ax = plt.subplots()
        # ax.scatter(Z_tsne[:,0], Z_tsne[:,1], alpha=0.5)
        # plt.ylabel('z2')
        # plt.xlabel('z1')
        # plt.title('t-SNE')
        # utils.savefig('tSNE_RNAseq.png', 'cancerGeneExpressionRNASeq')

    else:
        print("Unknown question: %s" % question)    