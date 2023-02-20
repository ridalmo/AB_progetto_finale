from astropy.io import fits 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def Covariance(matrix,averege = False):
    AVE = matrix.mean(axis = 0)
    dim_matrix = len(matrix[0,:])
    D = []
    
    for i in range(dim_matrix):
        D.append(matrix[:,i]-AVE[i])
    D = np.array(D)
    
    cov = np.matmul(D,D.T)
    
    if averege:
        return AVE, cov
    else:
        return np.array(cov)/(len(matrix[:,0])-1)
    
def Read(N_mis,test,keyword):
    
    measures = []
    for i in tqdm(np.arange(N_mis)+1,dynamic_ncols=True,desc='Reading: '):
        fname = f'..\data\MockMeasures_2PCF_Test{test}\MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'
        # print(f'open file N {i} in test : {test}')
        
        file = fits.open(fname)
        table = file[1].data.copy()
        measures.append(table[keyword])
        if (i==1):
            scale = table['SCALE']
        del table
        file.close()
    return np.array(measures), np.array(scale)

def Correlation(matrix):
    dim_matrix = matrix.shape
    diag = np.diag(matrix)
    COV = np.zeros(dim_matrix)
    for i in range(dim_matrix[0]):
        for j in range(dim_matrix[1]):
            COV[i,j]=matrix[i,j]/np.sqrt(diag[i]*diag[j])
    return COV


# to do :
# theretical covarince

def Th_Covariance(distance,multipoli, params):
    l_1 = multipoli[0]
    l_2 = multipoli[1]
    somma_h = params[l_1]['h']**2+params[l_2]['h']**2
    dim = len(distance)
    COV = np.zeros((dim,dim))
    if (multipoli[0]==multipoli[1]):
        for i in range(dim):
            for j in range(dim):
                COV[i,j] = (params[l_1]['sigma']**2)*np.exp(-((distance[i]-distance[j])**2)/(2*params[l_1]['h']**2))
        return COV
    elif (multipoli[0] != multipoli[1]):
        for i in range(dim):
            for j in range(dim):
                COV[i,j] = (params[l_1]['sigma']*params[l_2]['sigma'])*np.sqrt((2*params[l_1]['h']*params[l_2]['h'])/(somma_h))*np.exp(-((distance[i]-distance[j])**2)/somma_h)
        return COV

def Res(th_covariance,covariance,N_mis,N_bins = 200):
    TH_CORR = Correlation(th_covariance)
    RES = np.zeros((N_bins,N_bins))
    diag = np.diag(th_covariance)
    for i in range(N_bins):
        for j in range(N_bins):
            RES[i,j] = (th_covariance[i,j] - covariance[i,j])*np.sqrt((N_mis-1)/((1+TH_CORR[i,j])*diag[i]*diag[j]))
    return RES

# function for fancy plot

def plot_figure(array_matrix, cmap = 'magma',size = (7,5),title = ''):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.imshow(array_matrix,cmap=cmap)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical',ax = ax)
    plt.show()
        
    return

def plot_figures(array_matrix1,array_matrix2, cmap = 'magma',size = (7,5),title_1 = '',title_2 = ''):
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(1,2, wspace = 0.1)
    axs = gs.subplots(sharey=True)
    map1 = axs[0].imshow(array_matrix1,cmap = cmap)
    map2 = axs[1].imshow(array_matrix2,cmap = cmap)

    axs[0].set_title(title_1, fontsize = 20)
    axs[1].set_title(title_2, fontsize = 20)
    axs[0].set_xlabel('N Bins',fontsize = 16)
    axs[1].set_xlabel('N Bins',fontsize = 16)
    axs[0].set_ylabel('N Bins',fontsize = 16)
    # cax = fig.add_axes([0.1, 0.1, 0.8, 0.3])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.set_frame_on(False)
    cbar1 = plt.colorbar(map1,orientation = 'horizontal',ax=axs[1])
    cbar2 = plt.colorbar(map2,orientation = 'horizontal',ax=axs[0])
    
    cbar1.ax.tick_params(labelsize = 12)
    cbar2.ax.tick_params(labelsize = 12)
    return

# function for cross-corr matrix

def Cross_covariance(matrix):
    D = [[],[],[]]
    cov = []
    for j in range(len(matrix)):
        AVE = matrix[j].mean(axis = 0)
        dim_matrix = len(matrix[j][0,:])
        
        for i in range(dim_matrix):
            D[j].append(matrix[j][:,i]-AVE[i])
        D[j] = np.array(D[j])
        del AVE
    for k in range(3):
        for l in range(3):
            cov.append(np.matmul(D[k],D[l].T)/(len(matrix[0][:,0])-1))
    
    return np.array(cov)

def block_matrix(matrix):
    COVV2 = np.concatenate(matrix[0:3])
    COVV3 = np.concatenate(matrix[3:6])
    COVV4 = np.concatenate(matrix[6:9])
    
    return np.concatenate((COVV2,COVV3,COVV4),axis=1)

def Res_cross(th_matrix,matrix,N_mis,N_bins = 200):
    RES = []
    for k in range(9):
        RES.append(Res(th_matrix[k],matrix[k],N_mis))
        
    return np.array(RES)

def Th_Cross_Covariance(matrix,params):
    theoretical_cross_correlation = []
    for i in range(3):
        for j in range(3):
            theoretical_cross_correlation.append(Th_Covariance(matrix[i],[i,j],params))
        
    return np.array(theoretical_cross_correlation)
