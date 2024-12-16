from scipy.stats import multivariate_normal
import random
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import scipy as sc

import warnings
warnings.filterwarnings("ignore")

random.seed(17)
POS_LABEL = 1
NEG_LABEL = 0


def Entropy(P):
    H = P*np.log2(P)+(1-P)*np.log2(1-P)
    H *= -1
    return H

def getW(clf):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    b = - (clf.intercept_[0]) / w[1]

    W = np.hstack([a,b])
    return W

def prepare_data(nTrain,nCand):
    mu_first = [1, 1]
    mu_second = [0.0, 1.3]  # [-1, -1]
    cov_first = [[0.06, 0.001], [0.001, 0.025]]
    cov_second = [[0.07, 0.001], [0.001, 0.02]]

    random.seed(17)

    mn = multivariate_normal(mean=mu_first, cov=cov_first)
    out=mn.rvs(size=(nTrain + nCand) // 5, random_state=12345)
    x1 = out[:,0]
    y1 = out[:,1]

    mn = multivariate_normal(mean=mu_second, cov=cov_second)
    out = mn.rvs(size=4*(nTrain + nCand) // 5, random_state=12345)
    x2 = out[:, 0]
    y2 = out[:, 1]

    X1 = np.hstack([x1.reshape(-1, 1), y1.reshape(-1, 1)])
    Y1 = np.ones([X1.shape[0], 1])
    X2 = np.hstack([x2.reshape(-1, 1), y2.reshape(-1, 1)])
    Y2 = np.zeros([X2.shape[0], 1])

    X = np.vstack([X1, X2])
    Y = np.vstack([Y1, Y2])
    idx = np.arange(X.shape[0])
    random.Random(4).shuffle(idx)

    X = X[idx, :]
    Y = Y[idx]

    x_train = X[0:nTrain, :]
    y_train = Y[0:nTrain]

    x_cand = X[nTrain:-1, :]
    y_cand = Y[nTrain:-1]

    return x_train,y_train, x_cand, y_cand

def prepare_line(X,Y,y_min,y_max,clf,xcand,ycand, sel_idx=None, clf_gt=None, clf_start=None):


    clf.fit(X, Y)
    w = clf.coef_[0]
    a = -w[0] / w[1]
    b = - (clf.intercept_[0]) / w[1]
    #b=0
    yline = np.linspace(y_min, y_max)
    xline = (yline-b)/a

    P = clf._predict_proba_lr(xcand)
    P = P[:,1]
    ap = average_precision_score(ycand, P)

    delta_W = 0
    if clf_gt is not None:
        w_gt = clf_gt.coef_[0]
        a_gt = -w_gt[0] / w_gt[1]
        b_gt = - (clf_gt.intercept_[0]) / w_gt[1]
        delta_W = np.linalg.norm(a-a_gt)+np.linalg.norm(b-b_gt)

    Div, uncertainty = 0,0
    if sel_idx is not None and clf_start is not None:
        xt = xcand[sel_idx,:]
        Dist = cdist(xt, xt)
        mx_dist = (np.max(Dist))
        ind = 0
        M = 0
        for i in range(xt.shape[0]):
            for j in range(i+1,xt.shape[0]):
                ind += 1
                M += Dist[i,j]
        M = M/ind
        Div = M

        P = clf_start._predict_proba_lr(xt)
        P = P[:, 1]
        #entropyVec = -(1.0 * (P * (np.log(P)) + (1.0 - P) * np.log(1.0 - P)))
        entropyVec = Entropy(P)
        uncertainty = np.mean(entropyVec)

    return xline,yline,ap,delta_W,Div,uncertainty

def AL_rand(xcand, B):
    N = xcand.shape[0]
    arr = np.arange(N)
    random.shuffle(arr)
    idx = arr[0:B]
    return idx



def AL_rand0(x_train, y_train, x_cand, clf, B):
    N = x_cand.shape[0]
    arr = np.arange(N)
    random.shuffle(arr)
    idx = arr[0:B]
    return idx

def add_point(x_train, y_train, x_cand, y_cand):

    y_train = np.reshape(y_train,(-1,1))
    xcandc = np.vstack([x_train, np.asarray(x_cand)])
    new_y = np.asarray(y_cand)
    ycandc = np.vstack([y_train, new_y])
    #ycandc = np.vstack([np.asarray(y_train), new_y])
    return xcandc, ycandc

def remove_points(xcand,ycand,idx):
   N = np.arange(xcand.shape[0])
   active_idx = np.array(list(set(N)-set(idx)))
   new_xcand = xcand[active_idx,:]
   new_ycand = ycand[active_idx,:]
   return new_xcand, new_ycand



def AL_GAL_next_point(x_train, y_train, x_cand, ignore_idx, clf):

    neg_label = NEG_LABEL
    pos_label = POS_LABEL
    slope_only=False

    #if len(set(list(y_train))) <= 1:


    clf.fit(x_train, y_train)
    if slope_only:
        W0 = clf.coef_[0]
    else:
        W0 = getW(clf)


    N = x_cand.shape[0]
    vecM = np.zeros((N))

    Mminus = np.zeros(N)
    Mplus = np.zeros(N)
    Mpred = np.zeros(N)
    for i in range(N):
        if i in ignore_idx:
            continue
        y_train = np.reshape(y_train,(-1,1))
        xc = np.reshape(x_cand[i, :], (1, -1))
        xt, yt = add_point(x_train, y_train, xc, neg_label)
        clf.fit(xt, np.squeeze(yt))
        if slope_only:
            W_minus = clf.coef_[0]
            m_minus = np.linalg.norm(W0 - W_minus)
        else:
            W_minus = getW(clf)
            m_minus = np.linalg.norm(W0[0] - W_minus[0])+ np.linalg.norm(W0[1] - W_minus[1])
        Mminus[i] = m_minus
        xt, yt = add_point(x_train, y_train, xc, pos_label)
        clf.fit(xt, np.squeeze(yt))
        if slope_only:
            W_plus = clf.coef_[0]
            m_plus = np.linalg.norm(W0 - W_plus)
        else:
            W_plus = getW(clf)
            m_plus = np.linalg.norm(W0[0] - W_minus[0]) + np.linalg.norm(W0[1] - W_minus[1])

        Mplus[i] = m_plus
        vecM[i] = min(m_minus, m_plus)
        if m_minus < m_plus:
            Mpred[i] = neg_label

        else:
            Mpred[i] = pos_label

    idx = np.argmax(vecM)
    return idx, Mpred[idx],vecM[idx]

def AL_GAL(xtrain, ytrain, xcand, clf, B):
    # if len(ytrain.shape) == 1:
    #     #ytrain = ytrain.ravel()
    #     ytrain=ytrain.reshape(-1,1)
    ignore_idx = []
    scores = []
    xt = xtrain
    yt = ytrain
    for i in range(B):
        idx,pl, M = AL_GAL_next_point(xt, yt, xcand, ignore_idx, clf)
        #print(idx)
        xt, yt = add_point(xt, yt, xcand[idx], pl)
        ignore_idx.append(idx)
        scores.append(M)


    return ignore_idx,scores

def AL_entropy(xtrain, ytrain, xcand, clf, B):
    clf.fit(xtrain, ytrain)
    P = clf._predict_proba_lr(xcand)
    P = P[:,1]
    #entropyVec = -( 1.0 * (P * (np.log2(P)) + (1.0 - P) * np.log2(1.0 - P)) )
    entropyVec = Entropy(P)
    idx = np.argsort(entropyVec)[::-1][:B]


    return idx


def kmeans_plus_batch(xcand, K):

    if K == 1:
        mean_vector = np.mean(xcand, axis=0)
        distances = np.linalg.norm(xcand - mean_vector, axis=1)
        return [np.argmin(distances)]
    else:
        kmeans = KMeans(init="k-means++",
                    n_clusters=K,
                    n_init=15)

        kmeans.fit(xcand)
        centers = kmeans.cluster_centers_
        Dist = cdist(xcand, centers)
        indices = np.argmin(Dist,axis=0)
    indices = list(indices)
    return indices

def AL_diversity(xtrain, ytrain, xcand, clf, B):

    idx = kmeans_plus_batch(xcand, B)

    return idx


def AL_maxmin_interp(xtrain, ytrain, xcand, clf, B):

    pos_label = POS_LABEL
    neg_label = NEG_LABEL
    nCand = xcand.shape[0]
    vecM = np.zeros(nCand)
    for i in range(nCand):
        xc = xcand[i,:]
        xt, yt = add_point(xtrain, ytrain, xc, neg_label)
        if len(set(yt.squeeze())) > 1:
            clf.fit(xt, yt)
            W_minus = clf.coef_
        else:
            W_minus = np.mean(xt, axis=0)
        m_minus = np.linalg.norm(W_minus)
        xt, yt = add_point(xtrain, ytrain, xc, pos_label)
        if len(set(yt.squeeze())) > 1:
            clf.fit(xt, yt)
            W_plus = clf.coef_
        else:
            W_plus = np.mean(xt, axis=0)
        m_plus = np.linalg.norm(W_plus)
        vecM[i] = min(m_minus, m_plus)

    idx = np.argsort(vecM)[::-1][:B]
    return idx

def AL_ranked(xtrain, ytrain, xcand, clf, B):
        similarity_score = 0.2
        nUnlabeled = xcand.shape[0]
        nLabeled = xtrain.shape[0]
        uncertainty = 1-similarity_score
        dMatrix = sc.spatial.distance.cdist(xcand, xtrain)
        distance_scores = np.min(dMatrix, axis=1)
        similarity_scores = 1 / (1 + distance_scores)
        alpha = nUnlabeled / (nUnlabeled+nLabeled)
        scores = alpha * (1 - similarity_scores) + (1 - alpha) * uncertainty
        idx = np.argsort(scores)[::-1][:B]

        return idx

def AL_cod(xtrain, ytrain, xcand, clf, B):

    nCand = xcand.shape[0]
    if len(set(ytrain.squeeze())) <= 1:
        clf.coef_ = np.mean(xtrain, axis=0)
    else:
        clf.fit(xtrain, ytrain)
    W = clf.coef_.T
    Wprev = clf.coef_.T
    ynew = xcand.dot(W).reshape(1,nCand)
    yprev = xcand.dot(Wprev).reshape(1,nCand)
    deltay = np.abs(ynew-yprev).squeeze()
    idx = np.argsort(deltay)[::-1][:B]

    if len(set(ytrain.squeeze())) <= 1:
        clf.coef_ = np.mean(xtrain,axis=0)
    else:
        clf.fit(xtrain, ytrain)

    return idx

