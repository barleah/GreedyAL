import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
import random,copy,sklearn
import AL_utils

import warnings
warnings.filterwarnings("ignore")

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])






def main():
    print('start')

    # random.seed(11)
    # nTrain = 17  # 10
    # nCand = 150#150  # 150
    # YMIN = 0.3
    # YMAX = 2.0
    # B=5

    nTrain = 15  # 10
    nCand = 200 #240  # 150  # 150
    YMIN = 0.3
    YMAX = 2.0
    B = 6


    clf0 = LinearSVC(random_state=0,fit_intercept=True)
    clf_start = copy.deepcopy(clf0)
    clf_gt = copy.deepcopy(clf0)

    x_train, y_train, x_cand, y_cand = AL_utils.prepare_data(nTrain, nCand)
    clf_start.fit(x_train,y_train)


    X = np.vstack([x_train, x_cand])
    Y = np.vstack([y_train, y_cand])
    clf_gt.fit(X,Y)
    xline_gt, yline_gt,ap_gt,_ ,_,_= AL_utils.prepare_line(X, Y, YMIN, YMAX, clf0,x_cand,y_cand)
    idx_rand = AL_utils.AL_rand(x_cand,B)
    xline_start, yline_start,ap_start,_,_,_ = AL_utils.prepare_line(x_train,
                                                   y_train, YMIN, YMAX, clf0,x_cand, y_cand)




    xline_rand, yline_rand,ap_rand,dW_rand,div_rand,unc_rand = \
        AL_utils.prepare_line(np.vstack([x_train,x_cand[idx_rand,:]]),
                                                   np.vstack([y_train, y_cand[idx_rand]]),
                              YMIN, YMAX, clf0,x_cand,y_cand,idx_rand,clf_gt,clf_start)

    idx_gal,scores_gal = AL_utils.AL_GAL(x_train, y_train, x_cand, clf0, B)
    xline_gal, yline_gal,ap_gal,dW_gal,div_gal,unc_gal = \
        AL_utils.prepare_line(np.vstack([x_train, x_cand[idx_gal, :]]),
                                                  np.vstack([y_train, y_cand[idx_gal]]),
                              YMIN, YMAX, clf0,x_cand,y_cand,idx_gal,clf_gt,clf_start)

    idx_entropy = AL_utils.AL_entropy(x_train, y_train, x_cand, clf0, B)
    xline_ent, yline_ent, ap_ent,dW_ent,div_ent,unc_ent = AL_utils.prepare_line(np.vstack([x_train, x_cand[idx_entropy, :]]),
                                                 np.vstack([y_train, y_cand[idx_entropy]]),
                                                                                YMIN, YMAX,
                                                                                clf0,x_cand,y_cand,idx_entropy,clf_gt,clf_start)

    idx_div = AL_utils.AL_diversity(x_train, y_train, x_cand, clf0, B)

    xline_div, yline_div, ap_div, dW_div,div_div,unc_div = AL_utils.prepare_line(np.vstack([x_train, x_cand[idx_div, :]]),
                                                         np.vstack([y_train, y_cand[idx_div]]), YMIN, YMAX, clf0,
                                                         x_cand, y_cand,idx_div, clf_gt,clf_start)

    #-------------------------------
    plt.figure()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=70, cmap=cm_bright, edgecolors='k')
    plt.scatter(x_cand[:, 0], x_cand[:, 1], c=y_cand, s=70, cmap=cm_bright,
                edgecolors='k', alpha=0.2)
    plt.scatter(x_cand[idx_rand, 0], x_cand[idx_rand, 1], s=120, marker='o',
                facecolor='none', edgecolors='yellow', linewidths=3, label='selected points')
    plt.plot(xline_gt, yline_gt, "k-.", linewidth=2, label='classifier with all data')
    plt.plot(xline_start, yline_start, "y--", linewidth=3, label='classifier before selection')
    plt.plot(xline_rand, yline_rand, "y-", linewidth=3, label='classifier after selection')
    #plt.title('dW={:.2f} diversity={:.2f} uncertainty={:.2f}'.format(dW_rand,div_rand,unc_rand))
    plt.title('Random')
    plt.legend()

    #----------------------------------------------------------
    plt.figure()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=70, cmap=cm_bright, edgecolors='k')
    plt.scatter(x_cand[:, 0], x_cand[:, 1], c=y_cand, s=70, cmap=cm_bright,
                edgecolors='k', alpha=0.2)
    plt.scatter(x_cand[idx_gal, 0], x_cand[idx_gal, 1], s=120, marker='o',
                facecolor='none', edgecolors='forestgreen', linewidths=3, label='selected points')
    plt.plot(xline_gt, yline_gt, "k-.", linewidth=2, label='classifier with all data')
    plt.plot(xline_start, yline_start, c='forestgreen',linestyle='--', linewidth=3, label='classifier before selection')

    for jj in range(B):
        plt.text(x_cand[idx_gal[jj], 0]-0.06, x_cand[idx_gal[jj], 1] - 0.12, 'i{}'.format(jj),fontsize=11)
    plt.plot(xline_gal, yline_gal, c='forestgreen', linewidth=3, label='classifier after selection')
    #plt.title('dW={:.2f} diversity={:.2f} uncertainty={:.2f}'.format(dW_gal, div_gal, unc_gal))
    plt.title('GAL')
    plt.legend()

    #--------------------------------------------------------------------
    plt.figure()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=70, cmap=cm_bright, edgecolors='k')
    plt.scatter(x_cand[:, 0], x_cand[:, 1], c=y_cand, s=70, cmap=cm_bright,
                edgecolors='k', alpha=0.2)
    plt.scatter(x_cand[idx_entropy, 0], x_cand[idx_entropy, 1], s=120, marker='o',
                facecolor='none', edgecolors='magenta', linewidths=3, label='selected points')
    plt.plot(xline_gt, yline_gt, "k-.", linewidth=2, label='classifier with all data')
    plt.plot(xline_start, yline_start, "m--", linewidth=3, label='classifier before selection')

    plt.plot(xline_ent, yline_ent, "m-", linewidth=3, label='classifier after selection')
    #plt.title('dW={:.2f} diversity={:.2f} uncertainty={:.2f}'.format(dW_ent, div_ent, unc_ent))
    plt.title('Uncertainty: Entropy')
    plt.legend()

    #-------------------------------------------------------------------
    plt.figure()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=70, cmap=cm_bright, edgecolors='k')
    plt.scatter(x_cand[:, 0], x_cand[:, 1], c=y_cand, s=70, cmap=cm_bright,
                edgecolors='k', alpha=0.2)
    plt.scatter(x_cand[idx_div, 0], x_cand[idx_div, 1], s=120, marker='o',
                facecolor='none', edgecolors='cyan', linewidths=3, label='selected points')
    plt.plot(xline_gt, yline_gt, "k-.", linewidth=2, label='classifier with all data')
    plt.plot(xline_start, yline_start, "c--", linewidth=3, label='classifier before selection')
    plt.plot(xline_div, yline_div, "c-", linewidth=3, label='classifier after selection')
    #plt.title('dW={:.2f} diversity={:.2f} uncertainty={:.2f}'.format(dW_div, div_div, unc_div))
    plt.title('Diversity: Kmeans++')
    plt.legend()

    #
    #
    # ####
    #
    # xline_gal_plus, yline_gal_plus, ap_gal, dW_gal, div_gal, unc_gal = \
    #     AL_utils.prepare_line(np.vstack([x_train, x_cand[idx_gal[0], :]]),
    #                           np.vstack([y_train, 1]),
    #                           YMIN, YMAX, clf0, x_cand, y_cand, idx_gal, clf_gt, clf_start)
    #
    # xline_gal_minus, yline_gal_minus, ap_gal, dW_gal, div_gal, unc_gal = \
    #     AL_utils.prepare_line(np.vstack([x_train, x_cand[idx_gal[0], :]]),
    #                           np.vstack([y_train, 0]),
    #                           YMIN, YMAX, clf0, x_cand, y_cand, idx_gal, clf_gt, clf_start)
    # plt.figure()
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=70, cmap=cm_bright, edgecolors='k')
    # plt.scatter(x_cand[:, 0], x_cand[:, 1], c=y_cand, s=70, cmap=cm_bright,
    #             edgecolors='k', alpha=0.2)
    # plt.scatter(x_cand[idx_gal[0], 0], x_cand[idx_gal[0], 1], s=120, marker='o',
    #             facecolor='none', edgecolors='forestgreen', linewidths=3)
    #
    # plt.plot(xline_gt, yline_gt, "k-.", linewidth=2, label='classifier with all data')
    # plt.plot(xline_start, yline_start, c='forestgreen', linestyle='--', linewidth=3,
    #          label='classifier before selection')
    # plt.plot(xline_gal_plus, yline_gal_plus, "b-", linewidth=2, label='pseudo label = +1')
    # plt.plot(xline_gal_minus, yline_gal_minus, "r-", linewidth=2, label='pseudo label = -1')
    # plt.legend()
    #
    # plt.figure()
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=70, cmap=cm_bright, edgecolors='k')
    # plt.scatter(x_cand[:, 0], x_cand[:, 1], c=y_cand, s=70, cmap=cm_bright,
    #             edgecolors='k', alpha=0.2)
    # plt.scatter(x_cand[idx_gal, 0], x_cand[idx_gal, 1], s=120, marker='o',
    #             facecolor='none', edgecolors='forestgreen', linewidths=3, label='selected points')
    #
    # plt.plot(xline_gt, yline_gt, "k-.", linewidth=2, label='classifier with all data')
    # for jj in range(B):
    #     plt.text(x_cand[idx_gal[jj], 0] - 0.06, x_cand[idx_gal[jj], 1] - 0.12, 'i{}'.format(jj), fontsize=11)
    # plt.plot(xline_gal, yline_gal, c='forestgreen', linewidth=3, label='classifier after selection')
    # plt.legend()



    plt.show()


if __name__ == "__main__":
    main()