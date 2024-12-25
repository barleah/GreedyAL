import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.models import VGG16_Weights
import random,os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle,copy
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_auc_score
import AL_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg = vgg.to(device)
vgg.classifier = vgg.classifier[:-1]  # Remove the last layer for embeddings
vgg.eval()
seed = 42
random.seed(seed)

generator = torch.Generator().manual_seed(seed)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for VGG16
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx  # Add index to track images


def extract_features(dataset):
    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, generator=generator)
    features = []
    labels = []
    indices = []
    with torch.no_grad():

        for images, lbls, idxs in dataloader:
            images = images.to(device)
            embeddings = vgg(images)
            embeddings = embeddings.cpu().numpy()
            features.append(embeddings)
            labels.extend(lbls.numpy())
            indices.extend(idxs.numpy())

    features = np.vstack(features)
    return normalize(features), np.array(labels), indices


def image_retrieval(dataset, features, labels, indices, query_idx, metadata, K, method='random',):
    """
    Perform image retrieval with active learning.

    Parameters:
    - dataset: Indexed dataset
    - features: Extracted image features
    - labels: True labels of the dataset
    - indices: Dataset indices for the features
    - query_idx: Index of the query image in the dataset
    - metadata: Dictionary containing hyperparameters (e.g., 'n_positive_labels', 'n_negative_labels', 'iterations', etc.)
    - K: Number of top candidates to consider
    - method: Active learning method ('random', 'entropy', 'diversity', 'gal', or 'cod')

    Returns:
    - A list of top similar images from the dataset
    - AP vector across iterations
    """

    labeled_pool = [query_idx]  # Start with the query image as labeled
    same_label_indices = [idx for idx in indices if labels[idx] == labels[query_idx] and idx != query_idx]
    labeled_pool.extend(same_label_indices[:metadata['n_positive_labels']])  # Add up to 3 samples with the same label

    different_label_indices = [idx for idx in indices if labels[idx] != labels[query_idx]]
    labeled_pool.extend(different_label_indices[:metadata['n_negative_labels']])  #

    unlabeled_pool = [idx for idx in indices if idx not in labeled_pool]

    feedback = {idx: 1 for idx in labeled_pool if labels[idx] == labels[query_idx]}  # Mark same-label samples as 'similar'
    feedback.update({idx: 0 for idx in labeled_pool if labels[idx] != labels[query_idx]})  # Mark different-label samples as 'not similar'


    query_label = labels[query_idx]
    ap_vec = np.zeros(metadata['iterations'])
   
    svm = LinearSVC(random_state=10,fit_intercept=True)
    clf = copy.deepcopy(svm)

    for i in range(metadata['iterations']):
        print(f"\rIteration: [{i + 1}/{metadata['iterations']} ({(i + 1) / metadata['iterations'] * 100:.1f}%)]",
              end="")

        # Train an SVM on the current labeled pool
        X_train = features[labeled_pool]
        y_train = np.array([feedback[idx] for idx in labeled_pool])
        
        svm.fit(X_train, y_train)

        # Compute similarities for unlabeled images
        X_unlabeled = features[unlabeled_pool]
        y_unlabeled = np.zeros(len(unlabeled_pool))
        y_unlabeled[labels[unlabeled_pool] == query_label] = 1
        similarities = svm.decision_function(X_unlabeled)
        
        y_scores = svm._predict_proba_lr(X_unlabeled)[:,1]
        preck, sorted_indices = precision_at_k(y_unlabeled,  y_scores, metadata['precision_at_k'])

        ap_vec[i] = preck

        if K is not None:
            top_indices = np.argsort(similarities)[::-1][:K]  # Select top K most similar
            X_candidates = features[unlabeled_pool][top_indices]
        else:
            X_candidates = features[unlabeled_pool]



        if method == 'random':
            selected_indices = AL_utils.AL_rand(X_candidates, metadata['B'])
        elif method == 'entropy':
            selected_indices = AL_utils.AL_entropy(X_train, y_train, X_candidates, clf, metadata['B'])
        elif method == 'diversity':
            selected_indices = AL_utils.AL_diversity(X_train, y_train, X_candidates, clf, metadata['B'])
        elif method == 'gal':
            selected_indices, scores_gal = AL_utils.AL_GAL(X_train, y_train, X_candidates, clf, metadata['B'])
        elif method == 'cod':
            selected_indices = AL_utils.AL_cod(X_train, y_train, X_candidates, clf, metadata['B'])
        else:
            raise ValueError(f"Unknown method: {method}")

        
        for idx in selected_indices:
            # Simulate feedback: Similar if the label matches the query label
            simulated_feedback = 1 if labels[idx] == query_label else 0
            feedback[idx] = simulated_feedback
            labeled_pool.append(idx)

        # Update unlabeled pool
        unlabeled_pool = [idx for idx in unlabeled_pool if idx not in selected_indices]

        if len(unlabeled_pool) == 0:
            print("No more unlabeled images.")
            break

    # Retrieve the top similar images based on final SVM scores

    y_scores = svm._predict_proba_lr(features)[:,1]
    y_unlabeled = np.zeros(len(labels))
    y_unlabeled[labels == query_label] = 1
    preck, top_indices = precision_at_k(y_unlabeled,  y_scores, metadata['precision_at_k'])
    top_indices=top_indices[:metadata['top_results']]

    print(labels[top_indices])
    return [dataset[indices[idx]] for idx in top_indices],ap_vec


def precision_at_k(y_true, y_scores, k):
    """
    Calculates Precision@K for a single query.
    
    Parameters:
    - y_true: List or array of true binary relevance labels (1 for relevant, 0 for not relevant).
    - y_scores: List or array of predicted scores or probabilities.
    - k: The number of top items to consider.
    
    Returns:
    - Precision@K value.
    """
    # Sort by predicted scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    top_k = sorted_indices[:k]

    # Count relevant items in top K
    relevant_count = sum(y_true[i] for i in top_k)
    
    return relevant_count / k, sorted_indices


def plot_query_and_top_images(dataset, query_idx, top_images, title="Query and Top Retrieved Images"):
    """
    dataset: The dataset containing the images.
    query_idx: Index of the query image.
    top_indices: List of indices of the top retrieved images.
    title: Title of the plot.
    """
    try:
        top_indices = [idx[2] for idx in top_images]
        # Determine the number of images to display (query + top retrieved images)
        num_images = len(top_indices) + 1  # Include the query image
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        # Plot the query image first
        query_img, query_label, _ = dataset[query_idx]
        query_img = query_img.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        normalized_image = (query_img - query_img.min()) / (query_img.max() - query_img.min())  # Scale to [0, 1]
        normalized_image = (normalized_image * 255).astype(np.uint8)
        query_img = normalized_image

        axes[0].imshow(query_img)
        axes[0].axis('off')
        axes[0].set_title(f"Query\nLabel: {query_label}")

        # Plot the top retrieved images
        for i, idx in enumerate(top_indices):
            img, label, _ = dataset[idx]
            if label == query_label:
                color='green'
            else:
                color='red'            
            

            img = img.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
            
            normalized_image = (img - img.min()) / (img.max() - img.min())  # Scale to [0, 1]
            normalized_image = (normalized_image * 255).astype(np.uint8)

            rect = Rectangle(
                        (0, 0),             # Bottom-left corner (x, y)
                        img.shape[1]+3,     # Width of the rectangle
                        img.shape[0]+3,     # Height of the rectangle
                        linewidth=6,        # Thickness of the rectangle's edge
                        edgecolor=color,    # Color of the rectangle
                        facecolor='none'    # Transparent fill
                    )


            axes[i + 1].imshow(normalized_image)
            axes[i + 1].add_patch(rect)
            axes[i + 1].axis('off')


        # Set overall title
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
    except Exception as e:
        print(f"Error while plotting images: {e}")


def plot_average_precision(ap_vec_list, exp_list=None, prec_k=10):
    """
    ap_vec_list: List of AP vectors for different experiments.
    exp_list: List of experiment names.
    """
    try:
        num_experiments = len(ap_vec_list)
        auc = np.sum(ap_vec_list, axis=1)
        plt.figure(figsize=(8, 6))
        for i, ap_vec in enumerate(ap_vec_list):
            plt.plot(np.arange(1, len(ap_vec) + 1), ap_vec, marker='o', markersize=4, label=f'{exp_list[i]}: AUC={auc[i]:.2f}')
       
        plt.xlabel("Iterations")
        plt.ylabel("Precision@K")
        plt.title("Precision@{}".format(prec_k))
        plt.legend()
        plt.grid(True)
        
    except Exception as e:
        print(f"Error while plotting AP: {e}")


def save_features_to_pickle(features, labels, indices, file_name="features.pkl"):
    with open(file_name, "wb") as f:
        pickle.dump({"features": features, "labels": labels, "indices": indices}, f)
    print(f"Features saved to {file_name}")


def load_features_from_pickle(file_name="features.pkl"):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    print(f"Features loaded from {file_name}")
    return data["features"], data["labels"], data["indices"]    


if __name__ == "__main__":

    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    indexed_dataset = IndexedDataset(cifar10)

    features_file = 'cifar10_features.pkl'
    
    metadata = {
            'query_index': 200,
            'B': 3,
            'iterations': 7,
            'K': 200,
            'precision_at_k': 15,
            'top_results': 15,
            'n_positive_labels': 4,
            'n_negative_labels': 4,
        }


    # Extract features and labels for the dataset
    if os.path.exists(features_file):
        features, labels, indices = load_features_from_pickle(features_file)
    else:
        print('preparing vgg features...')
        features, labels, indices = extract_features(indexed_dataset)
        save_features_to_pickle(features, labels, indices, file_name=features_file)


    # Query image
    query_idx = metadata['query_index']
    query_img, query_label, _ = indexed_dataset[indices[query_idx]]
    print(f"Query Image Label: {labels[query_idx]}")
  


    # Run interactive retrieval
    top_images_gal,ap_vec_gal = image_retrieval(indexed_dataset, features, labels, indices, query_idx, metadata,
                                                K = metadata['K'], method='gal')
    top_images_cod, ap_vec_cod = image_retrieval(indexed_dataset, features, labels, indices, query_idx, metadata,
                                                 K=metadata['K'], method='cod')
    top_images_diversity,ap_vec_diversity = image_retrieval(indexed_dataset, features, labels, indices, query_idx,
                                                            metadata,K=metadata['K'], method='diversity')
    top_images_random,ap_vec_random = image_retrieval(indexed_dataset, features, labels, indices, query_idx, metadata,
                                                      K=metadata['K'], method='random')
    top_images_entropy,ap_vec_entropy = image_retrieval(indexed_dataset, features, labels, indices, query_idx, metadata,
                                                        K=metadata['K'],method='entropy')
   
    
       
    plot_query_and_top_images(indexed_dataset, indices[query_idx], top_images_gal, 
                              title="Query and Top {} Retrieved Images {}".format(metadata['top_results'],'gal'))
    plot_query_and_top_images(indexed_dataset, indices[query_idx], top_images_diversity,
                               title="Query and Top {} Retrieved Images {}".format(metadata['top_results'],'diversity'))
    plot_query_and_top_images(indexed_dataset, indices[query_idx], top_images_random,
                               title="Query and Top {} Retrieved Images {}".format(metadata['top_results'],'random'))
    plot_query_and_top_images(indexed_dataset, indices[query_idx], top_images_entropy,
                               title="Query and Top {} Retrieved Images {}".format(metadata['top_results'],'entropy'))
    plot_query_and_top_images(indexed_dataset, indices[query_idx], top_images_cod,
                              title="Query and Top {} Retrieved Images {}".format(metadata['top_results'], 'cod'))
    plot_average_precision([ap_vec_gal,ap_vec_diversity,ap_vec_random,ap_vec_entropy,ap_vec_cod],['gal','diversity','random','entropy','cod'],metadata['precision_at_k'])
    plt.show()