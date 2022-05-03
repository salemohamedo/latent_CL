from task2vec.task2vec import Task2Vec
from task2vec.models import get_model
from task2vec import datasets
from task2vec import task_similarity

dataset_names = ('cifar10', 'cifar100')
# Change `root` with the directory you want to use to download the datasets
dataset_list = [datasets.__dict__[name](root='./data')[0] for name in dataset_names]

embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"Embedding {name}")
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets)+1)).cuda()
    embeddings.append( Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset) )

task_similarity.plot_distance_matrix(embeddings, dataset_names)
