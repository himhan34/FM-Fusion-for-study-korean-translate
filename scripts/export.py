import os, json
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.15, pad=0.05, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
            #  rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def compute_variance(association_mat, openset_names, min_rate=0.1):
    K = len(openset_names)
    # var = np.std(association_mat, axis=1)
    var = np.zeros(K)
    for k_ in np.arange(K):
        valid = association_mat[k_,:] > min_rate
        var[k_] = valid.sum()
    return var


def class_model(input_data,output_folder,min_rate=0.1,sum_rate=0.2):
    association_count, openset_id2name, nyu20names = input_data #np.load(input_data,allow_pickle=True)
    openset_name_list = [openset_id2name[i] for i in range(len(openset_id2name))]
    # print(openset_name_list)
    
    # hardcode floor detection
    if 'floor' in openset_name_list:
        floor_id = openset_name_list.index('floor')
        num_floors_detect = association_count[floor_id,:].sum()
        association_count[:,1] = 0 
        association_count[floor_id,:] = 0
        assert association_count[floor_id,:].sum() == 0
        association_count[floor_id,1] = num_floors_detect
    
    gt_cond_probability = normalize(association_count,axis=0,norm='l1') 
    det_cond_probability = normalize(association_count,axis=1,norm='l1')
    # IGNORE_NAMES = ['room','living room room','bathroom','bedroom','kitchen','appliance','stool']
    IGNORE_NAMES = ['stool','appliance','dish wash','blanket']
    HARDCODE_NAMES = ['infant bed']
    
    print('association range: {},{}'.format(gt_cond_probability.min(),gt_cond_probability.max()))
    
    openset_ids = [i for i in range(len(openset_id2name))]
    nyu20_ids = [i for i in range(len(nyu20names))]
    nyu20_name_legends = [name[:4] for name in nyu20names]
    
    # Filter invalid openset types
    MIN_COUNT=10
    super_name = np.count_nonzero(association_count,axis=1) == 1
    print('---- {} supernames----'.format(super_name.sum()))
    # for idx in openset_id2name:
    #     if super_name[idx]:
    #         gt_id = association_count[idx,:].argmax()
    #         assert association_count[idx,gt_id]==association_count[idx,:].sum()
    #         print('{}: {}, {} apperance'.format(openset_id2name[idx],nyu20names[gt_id],association_count[idx,:].sum()))
    
    # valid_rows = association_count.sum(axis=1) > 200
    valid_rows = (gt_cond_probability.max(axis=1) > min_rate) & (association_count.sum(axis=1) > MIN_COUNT)
    valid_names = np.ones(len(openset_id2name),np.bool_)
    hardcode_names = np.zeros(len(openset_id2name),np.bool_)
    for idx in openset_id2name:
        if openset_id2name[idx] in IGNORE_NAMES:
            valid_names[idx] = False
        if openset_id2name[idx] in HARDCODE_NAMES:
            hardcode_names[idx] = True
    # print('{}/{} valid'.format(valid_names.sum(),len(openset_id2name)))

    valid_rows = valid_rows & valid_names
    valid_rows = valid_rows | hardcode_names
    valid_openset = [openset_id2name[i] for i in np.where(valid_rows)[0].astype(np.int32)]
    valid_o_lengends = valid_openset #[openset_name_list[i,:5] for i in valid_rows]
    print('{}/{} openset types with rate>{}'.format(np.sum(valid_rows), len(openset_id2name),min_rate))
    valid_associations_count = association_count[valid_rows,:]
    # gt_cond_probability = gt_cond_probability[valid_rows,:]
    det_cond_probability = det_cond_probability[valid_rows,:]
    gt_cond_probability = normalize(valid_associations_count,axis=0,norm='l1')
    
    # print(nyu20names)
    assert len(nyu20_ids) == gt_cond_probability.shape[1]
    
    ## Baseline from Fusion++
    empirical_probability = np.zeros((len(valid_openset),len(nyu20names))) + 0.1
    empirical_association = json.load(open('benchmark/output/categories.json','r'))
    objects = empirical_association['objects']
    for gt_id, gt_name in enumerate(nyu20names):
        if gt_name not in objects: continue
        openset_names = objects[gt_name]['main']
        for openset in openset_names:
            if openset not in valid_openset: continue
            openset_id = valid_openset.index(openset)
            empirical_probability[openset_id,gt_id] = 0.9
    
    # association count
    fig, ax = plt.subplots(figsize=(8,8))
    im, cbar = heatmap(np.log(valid_associations_count), valid_o_lengends,nyu20_name_legends, ax=ax,
                       cmap='YlGn', cbarlabel='log(count)')
    ax.set(title='association count',xlabel='NYU_Set', ylabel='OpenSet({} types)'.format(len(valid_openset)))
    plt.savefig(os.path.join(output_folder,'association_count.png'))
    
    #todo: log likelihood
    export_name = 'likelihood'
    # export_name ='likelihood_log'
    # gt_cond_probability = -1.0/(np.log(gt_cond_probability))
    # det_cond_probability = -1.0/(np.log(det_cond_probability))
    # empirical_association = -1.0/(np.log(empirical_probability))
    
    # Analysis
    # var = compute_variance(gt_cond_probability[valid_rows,:], valid_openset)
    # print('-- openset variance:')
    # for kk in np.arange(var.shape[0]):
    #     if var[kk]>2: status = 'confuse'
    #     elif var[kk]==2: status = 'ambiguous'
    #     else: status='good' 
    #     print('{}: {}, {}'.format(valid_openset[kk], var[kk],status))
    
    # Plot
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
    im, cbar = heatmap((gt_cond_probability), valid_o_lengends, nyu20_name_legends, ax=ax1,
                    cmap="YlGn", cbarlabel="p(z|l)")
    ax1.set(title="likelihood p(z|l)",xlabel='NYU_Set', ylabel='OpenSet({} types)'.format(len(valid_openset)))
    im, cbar = heatmap(det_cond_probability, valid_o_lengends, nyu20_name_legends, ax=ax2,
                cmap="YlGn", cbarlabel="p(z|l)")
    ax2.set(title='p(l|z)',xlabel='NYU_Set', ylabel='OpenSet({} types)'.format(len(valid_openset)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,'{}.png'.format(export_name)))    
    
    fig, ax3 = plt.subplots(figsize=(8,8))
    im, cbar = heatmap(empirical_probability, valid_o_lengends, nyu20_name_legends, ax=ax3,
                       cmap="YlGn", cbarlabel="p(z|l)")
    ax3.set(title='kimera likelihood',xlabel='NYU_Set', ylabel='OpenSet({} types)'.format(len(valid_openset)))
    plt.savefig(os.path.join(output_folder,'empirical_{}.png'.format(export_name)))    

    
    # priors
    priors = association_count[valid_rows,:].sum(axis=0)
    priors = priors / priors.sum()
    fig, ax = plt.subplots(figsize=(8,12))
    
    valid_count = association_count[valid_rows,:].sum(axis=0)
    invalid_count = association_count[~valid_rows,:].sum(axis=0)
    ax.bar(nyu20names,valid_count,color='b',label='valid openset')
    ax.bar(nyu20names,invalid_count,color='r',label='ignored openset',bottom=valid_count)

    ax.legend()
    ax.set_title('priors p(l)')
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="center")
    plt.savefig(os.path.join(output_folder,'valid_dets.png'))

    # save model
    out = (priors,valid_associations_count,gt_cond_probability,det_cond_probability,empirical_probability, valid_openset, nyu20names)
    np.save(os.path.join(output_folder,'{}.npy'.format(export_name)),out,allow_pickle=True)
    
    # Ambiguous types
    ignore_id =[]
    ignore_names = []
    for typename in IGNORE_NAMES:   
        if typename in openset_name_list:   
            ignore_id.append(openset_name_list.index(typename))
            ignore_names.append(typename)
    if len(ignore_id)>0:
        ignore_id = np.array(ignore_id)
        fig, ax4 = plt.subplots(figsize=(8,8))
        ambiguous_likelihood = association_count[ignore_id,:] 
        ambiguous_likelihood = normalize(ambiguous_likelihood,axis=1,norm='l1')
        im, cbar = heatmap(ambiguous_likelihood, ignore_names, nyu20_name_legends, ax=ax4,
                        cmap="YlGn", cbarlabel="p(l|z)")
        ax4.set(title='Ambiguous types',xlabel='NYU_Set', ylabel='OpenSet({} types)'.format(len(valid_openset)))
        plt.savefig(os.path.join(output_folder,'ambiguous_types.png'))
    
    return valid_openset, nyu20_name_legends, gt_cond_probability, valid_rows

def likelihood_matrix(probability,output_folder,model_name):
    openset_names = probability['rows']
    nyu20names = probability['cols']
    likelihood = probability['likelihood']
    
    fig, ax = plt.subplots(figsize=(8,10))
    im, cbar = heatmap(likelihood, openset_names, nyu20names, ax=ax,
                       cmap="YlGn", cbarlabel="p(z|l)")
    ax.set(title=model_name,xlabel='NYU_Set', ylabel='OpenSet({} types)'.format(len(openset_names)))
    plt.savefig(os.path.join(output_folder,'{}.png'.format(model_name)))
    np.save(os.path.join(output_folder,'{}.npy'.format(model_name)),(openset_names, nyu20names, likelihood),allow_pickle=True)

def class_model_new(input_data,output_folder):
    association_count, openset_id2name, nyu20names = np.load(input_data,allow_pickle=True)
    
    gt_cond_probability = normalize(association_count,axis=0,norm='l1') 
    det_cond_probability = normalize(association_count,axis=1,norm='l1')
    min_count = 3
    supername_rate = 0.5
    top_names = 20
    
    
    difficult_types = ['bookshelf','cabinet','door','counter']
    fig, axes = plt.subplots(1,4,figsize=(32,12))
    
    #
    for axid in range(len(axes)):
        typename = difficult_types[axid]
        obj_id = nyu20names.index(typename)
        spname_idx = det_cond_probability[:,obj_id]>supername_rate
        spnames = [openset_id2name[opense_id] for opense_id in openset_id2name if spname_idx[opense_id]]
        
        openset_idx = [opense_id for opense_id in openset_id2name if association_count[opense_id,obj_id]>min_count]
        openset_rate = gt_cond_probability[openset_idx,obj_id]
        srt_indices = np.argsort(openset_rate)[::-1]
        # print('openset rate:{}'.format(openset_rate))
        # print('openset sort indices:{}'.format(srt_indices))
        
        openset_idx = np.array(openset_idx)[srt_indices]
        openset_rate = openset_rate[srt_indices]
        openset_names = [openset_id2name[opense_id] for opense_id in openset_idx]
        correct_rate = det_cond_probability[openset_idx,obj_id]
        # print('sorted:{}'.format(openset_idx))

        x_tick = np.arange(len(openset_names))
        width = 0.4
        axes[axid].bar(x_tick,openset_rate,width,label='gt')
        axes[axid].bar(x_tick+width,correct_rate,width,label='det')
        axes[axid].set_xticks(x_tick+width,openset_names)
        axes[axid].grid(True)
        axes[axid].legend()
        axes[axid].set_title(typename)
    
        plt.setp(axes[axid].get_xticklabels(), rotation=60, ha="right")
    plt.savefig(os.path.join(output_folder,'topnames.png'))
    # print('bookshelf is detected as {} openset names,\n {}'.format(len(openset_matches),openset_matches))
    print('bookshelf super names:{}'.format(spnames))
    
def prompts_histogram(class_prompts,output_folder,door_det_prompt):
    
    top_prompts = 30
    gt_name = 'door'
    
    related_prompts = class_prompts[gt_name]
    hist_ = {}
    
    # All related prompts
    for prompt in related_prompts:
        if prompt not in hist_: hist_[prompt] = 0
        hist_[prompt] += 1
    
    n_p = len(hist_)
    print('{} has {} matches and {} unique prompts'.format(gt_name,len(related_prompts),n_p))
    
    labels = []
    hist_vec = np.zeros(n_p,dtype=np.int32)

    for k,count in hist_.items():
        labels.append(k)
        hist_vec[labels.index(k)] = count
    
    srt_indices = np.argsort(hist_vec)[::-1]
    labels = [labels[i] for i in srt_indices]
    hist_vec = hist_vec[srt_indices]
    
    ax_id = 1
    fig, axes = plt.subplots(1,2,figsize=(28,12))
    axes[ax_id].bar(np.arange(top_prompts),hist_vec[:top_prompts])
    axes[ax_id].set_xticks(np.arange(top_prompts),labels[:top_prompts])
    axes[ax_id].set_title('top {} related prompts'.format(top_prompts))
    axes[ax_id].grid(True)
    plt.setp(axes[ax_id].get_xticklabels(), rotation=60, ha="right")

    
    # 2. det prompts histogram
    ax_id = 0
    width = 0.1
    top_prompts = door_det_prompt['top_prompts']
    specific_detnames = door_det_prompt['valid_openset_names']
    det_prompt_matrix = door_det_prompt['hist_mat']
    
    assert det_prompt_matrix.shape[0] == len(specific_detnames), 'invalid det prompt matrix'
    assert det_prompt_matrix.shape[1] == len(top_prompts)+1, 'invalid det prompt matrix {}!={}'.format(len(specific_detnames),det_prompt_matrix.shape[1])
    top_prompts.append('sum')
    
    for col in range(det_prompt_matrix.shape[1]):
        axes[ax_id].bar(np.arange(len(specific_detnames))+width*col,det_prompt_matrix[:,col],width,label=top_prompts[col])
    axes[ax_id].set_xticks(np.arange(len(specific_detnames)),specific_detnames)
    axes[ax_id].set_title('top {} detections with {} prompts'.format(len(specific_detnames),len(top_prompts)-1))
    axes[ax_id].grid(True)
    axes[ax_id].legend()
    plt.setp(axes[ax_id].get_xticklabels(), rotation=60, ha="right")
    # Save
    plt.savefig(os.path.join(output_folder,'{}.png'.format(gt_name)))

def instance_histogram(class_names, class_hist,output_folder):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(class_names,class_hist)
    ax.grid(True)
    ax.set_ylabel('count')
    ax.set_title('instance histogram of {} classes'.format(len(class_names)))
    
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    plt.savefig(os.path.join(output_folder,'priors.png'))

def extract_scores_ious(dir, openset_names, nyu20names):
    with open(dir,'r') as f:
        pairs = json.load(f) # list of dict
        f.close()
        assert len(pairs)>0, 'No pairs found!'
        
        K_ = len(openset_names)
        J_ = len(nyu20names)
        
        MAX_COUNT = 2000
        scores = np.zeros((K_,J_,MAX_COUNT),dtype=np.float32)
        ious = np.zeros((K_,J_,MAX_COUNT),dtype=np.float32)
        all_ious = []
        
        test_ious = []
        test_j = 6
        
        for pair in pairs:
            gt_label_id = pair['pair'][0]
            openset_name = pair['pair'][1]
            if (openset_name not in openset_names) or gt_label_id<0: continue
            openset_id = openset_names.index(openset_name)
            count_scores = (scores[openset_id,gt_label_id,:]>0).sum()
            if count_scores<MAX_COUNT: 
                assert pair['confidence']>0.0, 'Invalid confidence value: {}'.format(pair['confidence'])
                scores[openset_id,gt_label_id,count_scores] = pair['confidence'] 
            count_ious = (ious[openset_id,gt_label_id,:]>0).sum()
            if count_ious<MAX_COUNT: 
                assert pair['iou']>0.0, 'Invalid iou value: {}'.format(pair['iou'])
                ious[openset_id,gt_label_id,count_ious] = pair['iou']
            
            all_ious.append(pair['iou'])
            
            # test
            if openset_id==0 and gt_label_id==test_j:
                test_ious.append(pair['iou'])
        
        # verify
        # assert len(test_ious)>0, 'No test iou found!'
        # test_ious = np.array(test_ious)
        # print(test_ious.shape[0])
        # assert test_ious.shape[0]==(ious[0,test_j,:]>0.0).sum(), 'iou dim not match {}!={}'.format(test_ious.shape[0],(ious[0,test_j,:]>0.0).sum())
        # assert np.abs(test_ious.sum() - ious[0,test_j,:].sum()) <1e-6, 'iou not match {},{}!'.format(test_ious.sum(),ious[0,test_j,:].sum())
        all_ious = np.array(all_ious)
        print('{}/{} pairs are valid'.format((all_ious>0.5).sum(),all_ious.shape[0]))
        
        full_rank = (scores>0.0).sum(axis=2)
        print('{}/{} pairs are full rank recorded'.format((full_rank==MAX_COUNT).sum(),full_rank.size))
        print('scores range:{},{}'.format(scores.min(),scores.max()))
        print('ious range:{},{}'.format(ious.min(),ious.max()))
        return scores, ious, all_ious
        
if __name__=='__main__':
    class_data = 'benchmark/output/association_matrix.npy'
    # pairs_data = 'benchmark/pair_measurements.json'
    output_folder = 'benchmark/output'
    ASSOCIATION_THRESHOLD = 0.1

    openset_names, nyu20names, association_matrix, valid_rows = class_model(class_data,output_folder,ASSOCIATION_THRESHOLD)
    
    # class_model_new(class_data,output_folder)
    
    exit(0)
    
    # valid_pair = association_matrix > ASSOCIATION_THRESHOLD
    sum_valid_rate = np.sum(association_matrix,axis=0)
    for i in np.arange(sum_valid_rate.shape[0]):
        print('{}: {}'.format(nyu20names[i],sum_valid_rate[i]))
    
    # construct score and iou matrix
    scores, ious, allious = extract_scores_ious(pairs_data, openset_names, nyu20names)
    
    # Compute mean and variance
    mask_scores = np.ma.array(scores,mask=scores<1e-6)
    scores_mu = mask_scores.mean(axis=2)
    scores_var = mask_scores.var(axis=2)
    # scores_mu[~valid_pair] = 0.0
    
    mask_ious = np.ma.array(ious,mask=ious<1e-6)
    ious_mu = mask_ious.mean(axis=2)
    ious_var = mask_ious.var(axis=2)
    # ious_mu[~valid_pair] = 0.0
    # ious_var[~valid_pair] = 0.0
    
    # ious for sofa
    sofa_ious = ious[:,5,:].squeeze()
    sofa_ious = np.ma.array(sofa_ious,mask=sofa_ious<1e-6)
    # sofa_ious = np.mean(sofa_ious,axis=1)[valid_pair[:,5]]
    # print(sofa_ious)
    
    #
    flat_ious = allious #ious.flatten()
    print('{}/{} valid ious'.format((flat_ious>1e-6).sum(),flat_ious.size))
    
    flat_ious = flat_ious # [flat_ious>1e-6]
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(flat_ious,bins=100)
    ax_hist.set_xlabel('iou')
    ax_hist.set_ylabel('count')
    ax_hist.set_title('Analysis {} valid instance pairs'.format(len(flat_ious)))
    plt.savefig(os.path.join(output_folder,'ious_hist.png'))
    
    # Plot scores
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
    im, cbar = heatmap(scores_mu, openset_names, nyu20names, ax=ax1,
                    cmap="YlGn", cbarlabel="p(z^s|l^c,z^c)")

    ax1.set(title="scores",xlabel='NYU_Set', ylabel='OpenSet')
    
    # im, cbar = heatmap(scores_var, openset_names, nyu20names, ax=ax2,
    #             cmap="PiYG", cbarlabel="p(z^s|l^c,z^c)")
    # ax2.set(title="variance",xlabel='NYU_Set', ylabel='OpenSet')
    
    # plt.savefig(os.path.join(output_folder,'scores.png'))
    
    # Plot ious
    # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
    im, cbar = heatmap(ious_mu, openset_names, nyu20names, ax=ax2,
                    cmap="YlGn", cbarlabel="p(z^s|l^c,z^c)")

    ax2.set(title="iou",xlabel='NYU_Set', ylabel='OpenSet')
    
    # im, cbar = heatmap(ious_var, openset_names, nyu20names, ax=ax2,
    #             cmap="PiYG", cbarlabel="p(z^s|l^c,z^c)")
    # ax2.set(title="variance",xlabel='NYU_Set', ylabel='OpenSet')
    
    # plt.savefig(os.path.join(output_folder,'matches.png'))
