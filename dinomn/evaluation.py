import sklearn.metrics
import wandb

import numpy as np
import pandas as pd

import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import time


def compare_two_labels(label_model, label_gt, return_IoU_matrix, debug=False):
    
    # get number of detected nuclei
    nb_nuclei_gt = np.max(label_gt)
    nb_nuclei_model = np.max(label_model)
    
    # catch the case of an empty picture in model and gt
    if nb_nuclei_gt == 0 and nb_nuclei_model == 0:
        if(return_IoU_matrix):
            return [0, 0, 1, np.empty(0)]     
        else:
            return [0, 0, 1]
    
    # catch the case of empty picture in model
    if nb_nuclei_model == 0:
        if(return_IoU_matrix):
            return [0, nb_nuclei_gt, 0, np.empty(0)]     
        else:
            return [0, nb_nuclei_gt, 0]
    
    # catch the case of empty picture in gt
    if nb_nuclei_gt == 0:
        if(return_IoU_matrix):
            return [nb_nuclei_model, 0, 0, np.empty(0)]     
        else:
            return [nb_nuclei_model, 0, 0]
    
    # build IoU matrix
    IoUs = np.full((nb_nuclei_gt, nb_nuclei_model), -1.0, dtype = np.float32)

    # calculate IoU for each nucleus index_gt in GT and nucleus index_pred in prediction    
    # TODO improve runtime of this algorithm
    for index_gt in range(1,nb_nuclei_gt+1):

        nucleus_gt = label_gt == index_gt
        number_gt = np.sum(nucleus_gt)

        for index_model in range(1,nb_nuclei_model+1):
            
            if debug:
                print(index_gt, "/", index_model)
            
            nucleus_model = label_model == index_model 
            number_model = np.sum(nucleus_model)
            
            same_and_1 = np.sum((nucleus_gt == nucleus_model) * nucleus_gt)
            
            IoUs[index_gt-1,index_model-1] = same_and_1 / (number_gt + number_model - same_and_1)
    
    # get matches and errors
    detection_map = (IoUs > 0.5)
    nb_matches = np.sum(detection_map)

    detection_rate = IoUs * detection_map
    
    nb_overdetection = nb_nuclei_model - nb_matches
    nb_underdetection = nb_nuclei_gt - nb_matches
    
    mean_IoU = np.mean(np.sum(detection_rate, axis = 1))
    
    if(return_IoU_matrix):
        result = [nb_overdetection, nb_underdetection, mean_IoU, IoUs]
    else:
        result = [nb_overdetection, nb_underdetection, mean_IoU]
    return result

def measures_at(threshold, IOU):
    matches = IOU > threshold
    
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)

    prec = TP / (TP + FP)

    rec = TP / (TP + FN)

    return f1, prec, rec, TP, FP, FN


def segmentation_report(imid, predictions, gt, intersection_ratio=0.1, report_obj='Micronuclei'):
    # start = time.time()
    pi = skimage.morphology.label(predictions)
    gti = skimage.morphology.label(gt)
    nb_overdetection, nb_underdetection, mean_IoU, IoUs = compare_two_labels(pi, gti, True, False)
    if IoUs.size == 0:
        prec = 0.0
        rec = 0.0
        # print(f'----- Segmentation Report for {report_obj} -----')
        print(f'{imid},{prec},{rec}')
        wandb.log({'Precision':prec, 'Recall':rec})
    else:
        f1, prec, rec, TP, FP, FN = measures_at(intersection_ratio, IoUs)
        # print(f'----- Segmentation Report for {report_obj} -----')
        print(f'{imid},{prec:.4f},{rec:.4f}')
        wandb.log({'Precision':prec, 'Recall':rec})

def get_assignment(C, gt):
    # Map token predictions to pixels (multiply by 8)
    Y,X = list(C[0]*8), list(C[1]*8)
    C = np.asarray([(a,b) for a,b in zip(X,Y)])
    
    # Get true coordinates
    R = np.asarray(gt[["x","y"]])
    
    # Find nearest neighbor
    D = scipy.spatial.distance_matrix(C,R)
    assignment = np.argmin(D, axis=1)
    return assignment


def prediction_report(imid, probabilities, gt, threshold, output_dir):    
    ground_truth = np.zeros_like(probabilities)
    for k,r in gt.iterrows():
        a = r.y // 8
        b = r.x // 8
        ground_truth[a,b] = 1

    predictions = probabilities > threshold
    
    # Precision-recall curve

    GT = ground_truth.flatten()
    PRED = probabilities.flatten()

    plt.figure(figsize=(6,6))
    display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(
        GT, PRED, name="Detector"#, plot_chance_level=True
    )
    _ = display.ax_.set_title("Precision-Recall curve")
    plt.savefig(f"{output_dir}/{imid}-prcurve.png")
    
    # Classification report
    
    report = sklearn.metrics.classification_report(GT, PRED > threshold)
    text_file = open(f"{output_dir}/{imid}-report.txt", "w")
    text_file.write(report)
    text_file.close()
    
    correct = predictions * ground_truth
    missing = ground_truth - correct
    extra = predictions - correct
    print("Total:",np.sum(ground_truth),"Correct:",np.sum(correct), "Missing:", np.sum(missing), "Extra:", np.sum(extra))
    
    results = {
        "probabilities": probabilities,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "correct": correct,
        "missing": missing,
        "extra": extra
    }
    
    # Detailed report
    
    gt["Status"] = ""
    assignment = get_assignment(np.where(results["correct"]), gt)
    gt.loc[gt.index.isin(assignment), "Status"] = "correct"

    assignment = get_assignment(np.where(results["missing"]), gt)
    gt.loc[gt.index.isin(assignment), "Status"] = "missing" 
    
    gt.to_csv(f"{output_dir}/{imid}-details.csv", index=False)
    
    return results


def display_detections(im, imid, results, output_dir):
    # Show image
    fig, ax = plt.subplots(figsize=(30,30))
    ax.imshow(im)

    annotations = []

    # Display micronucleus boxes
    C = np.where(results["correct"])
    w,h = 16,16
    for i in range(len(C[0])):
        x1 = C[1][i]*8 - w
        y1 = C[0][i]*8 - h
        rect = patches.Rectangle((x1, y1), 2*w, 2*h, linewidth=1, edgecolor='gold', facecolor='none')
        ax.add_patch(rect)

    # Display micronucleus boxes
    C = np.where(results["missing"])
    w,h = 12,12
    for i in range(len(C[0])):
        x1 = C[1][i]*8 - w
        y1 = C[0][i]*8 - h
        rect = patches.Rectangle((x1, y1), 2*w, 2*h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, i, color="r", fontsize="xx-large")
        annotations.append({"col":C[1][i]*8, "row":C[0][i]*8, "ID":i, "color":"red","question":"Missed?"})

    # Display micronucleus boxes
    C = np.where(results["extra"])
    w,h = 12,12
    for i in range(len(C[0])):
        x1 = C[1][i]*8 - w
        y1 = C[0][i]*8 - h
        rect = patches.Rectangle((x1, y1), 2*w, 2*h, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, i, color="b", fontsize="xx-large")
        annotations.append({"col":C[1][i]*8, "row":C[0][i]*8, "ID":i, "color":"blue","question":"Real?"})

    plt.axis('off')
    plt.show()
    plt.savefig(f"{output_dir}/{imid}-fig.png")
    
    df = pd.DataFrame(annotations)
    df["answer"] = ""
    df = df.sort_values(by=["ID","question"])
    df.to_csv(f"{output_dir}/{imid}-checks.csv", index=False)

