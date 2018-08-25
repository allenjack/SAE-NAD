from argparse import ArgumentParser
import heapq
import numpy as np
import scipy

from code import eval_metrics, dataset
from code.model import AutoEncoder

import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


parser = ArgumentParser(description="SAE-NAD")
parser.add_argument('-e', '--epoch', type=int, default=60, help='number of epochs for GAT')
parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('--alpha', type=float, default=2.0, help='the parameter of the weighting function')
parser.add_argument('--epsilon', type=float, default=1e-5, help='the parameter of the weighting function')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('-att', '--num_attention', type=int, default=20, help='the number of dimension of attention')
parser.add_argument('--inner_layers', nargs='+', type=int, default=[200, 50, 200], help='the number of latent factors')
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5, help='the dropout probability')
parser.add_argument('-seed', type=int, default=0, help='random state to split the data')
args = parser.parse_args()


def get_mini_batch(X, weight_mask, batch_user_index):
    batch_item_index = []
    for user_id in batch_user_index:
        batch_item_index.append(X.getrow(user_id).indices)
    return X[batch_user_index].toarray(), weight_mask[batch_user_index].toarray(), batch_item_index


def log_surplus_confidence_matrix(B, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * np.log(1 + B / epsilon)
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S


def train_autoencoder(train_matrix, test_set):
    num_users, num_items = train_matrix.shape
    weight_matrix = log_surplus_confidence_matrix(train_matrix, alpha=args.alpha, epsilon=args.epsilon)
    train_matrix[train_matrix > 0] = 1.0
    place_correlation = scipy.sparse.load_npz('./data/Foursquare/place_correlation_gamma60.npz')

    assert num_items == place_correlation.shape[0]
    print(train_matrix.shape)

    # Construct the model by instantiating the class defined in model.py
    model = AutoEncoder(num_items, args.inner_layers, num_items, da=args.num_attention, dropout_rate=args.dropout_rate)
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss(size_average=False, reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.batch_size
    user_indexes = np.arange(num_users)

    model.train()
    for t in range(args.epoch):
        print("epoch:{}".format(t))
        np.random.shuffle(user_indexes)
        avg_cost = 0.
        for batchID in range(int(num_users / batch_size)):
            start = batchID * batch_size
            end = start + batch_size

            batch_user_index = user_indexes[start:end]

            batch_x, batch_x_weight, batch_item_index = get_mini_batch(train_matrix, weight_matrix, batch_user_index)
            batch_x_weight += 1
            batch_x = Variable(torch.from_numpy(batch_x).type(T.FloatTensor), requires_grad=False)

            y_pred = model(batch_item_index, place_correlation)

            # Compute and print loss
            batch_x_weight = Variable(torch.from_numpy(batch_x_weight).type(T.FloatTensor), requires_grad=False)
            loss = (batch_x_weight * criterion(y_pred, batch_x)).sum() / batch_size

            print(batchID, loss.data)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss / num_users * batch_size

        print("Avg loss:{}".format(avg_cost))

        # print the prediction score for the user 0
        print(model(
            [train_matrix.getrow(0).indices],
            place_correlation)[:, T.LongTensor(train_matrix.getrow(0).indices.astype(np.int32))])
        print(model([train_matrix.getrow(0).indices], place_correlation))

    # Evaluation
    model.eval()
    topk = 20
    recommended_list = []
    for user_id in range(num_users):
        user_rating_vector = train_matrix.getrow(user_id).toarray()
        pred_rating_vector = model([train_matrix.getrow(user_id).indices], place_correlation)
        pred_rating_vector = pred_rating_vector.cpu().data.numpy()
        user_rating_vector = user_rating_vector[0]
        pred_rating_vector = pred_rating_vector[0]
        pred_rating_vector[user_rating_vector > 0] = 0

        item_recommended_dict = dict()
        for item_inner_id, score in enumerate(pred_rating_vector):
            item_recommended_dict[item_inner_id] = score

        sorted_item = heapq.nlargest(topk, item_recommended_dict, key=item_recommended_dict.get)
        recommended_list.append(sorted_item)

        print(test_set[user_id], sorted_item[:topk])
        print(pred_rating_vector[sorted_item[0]], pred_rating_vector[sorted_item[1]],
              pred_rating_vector[sorted_item[2]], pred_rating_vector[sorted_item[3]],
              pred_rating_vector[sorted_item[4]])
        print("user:%d, precision@5:%f, precision@10:%f" % (
            user_id, eval_metrics.precision_at_k_per_sample(test_set[user_id], sorted_item[:5], 5),
            eval_metrics.precision_at_k_per_sample(test_set[user_id], sorted_item[:topk], topk)))

    precision, recall, MAP = [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(eval_metrics.precision_at_k(test_set, recommended_list, k))
        recall.append(eval_metrics.recall_at_k(test_set, recommended_list, k))
        MAP.append(eval_metrics.mapk(test_set, recommended_list, k))

    print(precision)
    print(recall)
    print(MAP)


def main():
    train_matrix, test_set, place_coords = dataset.Foursquare().generate_data(args.seed)
    train_autoencoder(train_matrix, test_set)


if __name__ == '__main__':
    main()
