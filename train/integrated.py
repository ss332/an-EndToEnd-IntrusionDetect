import numpy as np
from Model import model_v2

import torch
import torch.nn.functional as F
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from matplotlib.colors import LinearSegmentedColormap

from captum.attr import (
    IntegratedGradients,
    visualization
)

answer_words = ['Benign', 'FTP-BruteForce', 'SSH-Bruteforce', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris',
                'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-LOIC-UDP',
                'DDOS attack-HOIC', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection',
                'Infilteration',
                'Bot']

ids = [11217, 22432, 29953, 43350, 67527, 50970]
ids2 = [2430, 4218, 24732, 28923, 29890, 31629, 27442, 30203, 32629]
ids3 = [281, 8423, 13386, 15187, 17086, 27665]

ids45 = [21928, 35504, 48418, 77, 90, 154, 2300, 53075, 91768]
ids6 = [362, 372, 383, 991, 1043, 1007, 1391, 1402, 1415]

ids78 = [15424, 38895, 88287, 17968, 47267, 92242]

from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

arc_net = model_v2.ArcNet()
arc_net = ModelInputWrapper(arc_net)
arc_net = torch.nn.DataParallel(arc_net)

partial_dict = torch.load('model3.pth')
state = arc_net.module.state_dict()
state.update(partial_dict)
arc_net.module.load_state_dict(state)
arc_net.to(device)
arc_net.eval()

torch.backends.cudnn.enabled = False

attr = IntegratedGradients(arc_net)


def encode_packets(session):
    l = session.size(0)
    vec = torch.zeros(32, 320)
    if l >= 32:
        l = 32
    for i in range(l):
        vec[i] = session[i]
    return vec, torch.tensor(l)


def encode_sessions(session):
    session_tensor = torch.zeros(1024)
    size = 0
    for k in range(session.size(0)):
        x = session[k]
        for j in range(x.size(0)):
            if x[j] != 0 and size < 1024:
                session_tensor[size] = x[j]
                size = size + 1

        # (n,c,w,h)
    session_tensor = session_tensor.view(1, 1, 32, 32)
    return session_tensor


default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)


def gen_questions(l):
    questions = ""
    for i in range(l):
        questions = questions + " p{}".format(i)

    return questions


def arc_net_interpret(image_filename, target):
    session = torch.load(image_filename)

    img = encode_sessions(session)
    image_features = img.requires_grad_().to(device)
    original_image = img

    q, q_len = encode_packets(session)
    question = gen_questions(q_len)

    # generate reference for each sample
    q_input_embedding = q.unsqueeze(0)
    q_reference_baseline = torch.zeros((32, 320)).unsqueeze(0)

    inputs = (image_features, q_input_embedding)
    baselines = (image_features * 0.0, q_reference_baseline)

    ans = arc_net(*inputs, q_len.unsqueeze(0))

    # Make a prediction. The output of this prediction will be visualized later.
    pred, answer_idx = ans.topk(1)

    attributions = attr.attribute(inputs=inputs,
                                  baselines=baselines,
                                  target=answer_idx,
                                  additional_forward_args=q_len.unsqueeze(0),
                                  n_steps=100)

    # Visualize text attributions
    text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()

    print(attributions[1].sum(dim=2))
    print(attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm)
    print(answer_words[answer_idx], ',', target)
    print(question.split())
    score = (attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm).sum(0)
    vis_data_records = [visualization.VisualizationDataRecord(
        attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm,
        pred[0].item(),
        answer_words[answer_idx],
        answer_words[answer_idx],
        target,
        attributions[1].sum(),
        question.split(' '),
        0.0)]
    # visualization.visualize_text(vis_data_records)

    # visualize image attributions
    original_im_mat = np.transpose(original_image.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attributions_img = np.transpose(attributions[0].squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    visualization.visualize_image_attr_multiple(attributions_img, original_im_mat,
                                                ["original_image", "heat_map"], ["all", "absolute_value"],
                                                titles=["Session", 'attribution-' + answer_words[answer_idx]],
                                                cmap=default_cmap,
                                                show_colorbar=True)
    print('Text Contributions: ', score.item())
    print('Image Contributions: ', attributions[0].squeeze(0).sum().item())
    print('Total Contribution: ', attributions[0].sum().item() + score.item())
    return attributions


idx = 0  # cat
images = [r'C:\sessions\files8\session55351.pt',
          r'C:\sessions\files7\session38895.pt',
          r'C:\sessions\files7\session88287.pt']


def integrated_grads(filename):
    session = torch.load(filename)

    img = encode_sessions(session)
    image_features = img.requires_grad_().to(device)
    original_image = img

    q, q_len = encode_packets(session)
    question = gen_questions(q_len)

    # generate reference for each sample
    q_input_embedding = q.unsqueeze(0)
    q_reference_baseline = torch.zeros((32, 320)).unsqueeze(0)

    inputs = (image_features, q_input_embedding)
    baselines = (image_features * 0.0, q_reference_baseline)

    ans = arc_net(*inputs, q_len.unsqueeze(0))

    # Make a prediction. The output of this prediction will be visualized later.
    pred, answer_idx = F.softmax(ans, dim=1).data.cpu().max(dim=1)

    attributions = attr.attribute(inputs=inputs,
                                  baselines=baselines,
                                  target=answer_idx,
                                  additional_forward_args=q_len.unsqueeze(0),
                                  n_steps=100)

    return attributions


def cal_global_ig(f, cate):
    flows = pd.read_csv('../files/f_process{}.csv'.format(f))
    cate_df = flows[flows['label'] == cate]

    k = 30000
    print(k)
    packets_attri = torch.zeros(32)
    sessions_attri = torch.zeros((32, 32))

    for i in range(k):
        id = cate_df["id"].iloc[i]
        file = r'C:\sessions\files{}\session{}.pt'.format(f, id)
        attributions = integrated_grads(file)
        # arc_net_interpret(file,class_name(f,id))
        text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()
        packets_attri = packets_attri + attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm
        sessions_attri = sessions_attri + attributions[0].squeeze(0).squeeze(0)
        if i % 1000 == 0:
            print('progress:', i)

    packets_attri = packets_attri / k
    sessions_attri = sessions_attri / k
    id = cate_df["id"].iloc[i]
    file = r'C:\sessions\files{}\session{}.pt'.format(f, id)
    original_image = encode_sessions(torch.load(file))
    original_im_mat = np.transpose(original_image.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attributions_img = np.transpose(sessions_attri.unsqueeze(0).cpu().detach().numpy(), (1, 2, 0))

    visualization.visualize_image_attr_multiple(attributions_img, original_im_mat,
                                                ["original_image", "heat_map"], ["all", "absolute_value"],
                                                titles=["Session", 'attribution-' + answer_words[cate]],
                                                cmap=default_cmap,
                                                show_colorbar=True)
    torch.save(packets_attri, 'p{}.pt'.format(cate))
    torch.save(sessions_attri, 's{}.pt'.format(cate))
    print(cate)
    print(packets_attri)
    print(sessions_attri)


file = r'C:\sessions\files{}\session{}.pt'.format(5, 62)
attributions = integrated_grads(file)
a = attributions[1][0][0]
print(a.size())
c = []
for i in range(54):
    print(i + 1)
    print(a[i].item())
    if a[i].item() > 0:
        c.append(a[i].item())

print(c)
