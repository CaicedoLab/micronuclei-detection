import scipy
import sklearn.metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

