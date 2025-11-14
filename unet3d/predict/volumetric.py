import os
import torch
from monai.data import NibabelWriter
from monai.transforms import ResampleToMatch, LoadImage
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
# from tqdm import tqdm
from sklearn.decomposition import PCA
# import plotly.graph_objects as go
import numpy as np
from unet3d.utils.resample import resample_to_img
from unet3d.predict.utils import pytorch_predict_batch_array, get_feature_filename_and_subject_id
from unet3d.utils.utils import load_image
from unet3d.utils.one_hot import one_hot_image_to_label_map
import re


def load_volumetric_model(model_name, model_filename, n_gpus, strict, **kwargs):
    from unet3d.models.build import build_or_load_model
    model = build_or_load_model(model_name=model_name, model_filename=model_filename, n_gpus=n_gpus, strict=strict,
                                **kwargs)
    model.eval()
    return model


def load_images_from_dataset(dataset, idx, resample_predictions):
    if resample_predictions:
        x_image, ref_image = dataset.get_feature_image(idx, return_unmodified=True)
    else:
        x_image = dataset.get_feature_image(idx)
        ref_image = None
    return x_image, ref_image


def prediction_to_image(data, input_image, reference_image=None, interpolation="linear", segmentation=False,
                        segmentation_labels=None, threshold=0.5, sum_then_threshold=False, label_hierarchy=False):
    if data.dtype == torch.float16:
        data = torch.as_tensor(data, dtype=torch.float32)
    pred_image = input_image.make_similar(data=data)
    if reference_image is not None:
        pred_image = resample_to_img(pred_image, reference_image,
                                     interpolation=interpolation)
    if segmentation:
        pred_image = one_hot_image_to_label_map(pred_image,
                                                labels=segmentation_labels,
                                                threshold=threshold,
                                                sum_then_threshold=sum_then_threshold,
                                                label_hierarchy=label_hierarchy)
    return pred_image


def write_prediction_image_to_file(pred_image, output_template, subject_id, x_filename, prediction_dir, basename,
                                   verbose=False):
    if output_template is None:
        while type(x_filename) == list:
            x_filename = x_filename[0]
        pred_filename = os.path.join(prediction_dir,
                                     "_".join([subject_id,
                                               basename,
                                               os.path.basename(x_filename)]))
    else:
        pred_filename = os.path.join(prediction_dir,
                                     output_template.format(subject=subject_id))
    if verbose:
        print("Writing:", pred_filename)
    pred_image.to_filename(pred_filename)


def predict_volumetric_batch(model, batch, batch_references, batch_subjects, batch_filenames,
                             basename, prediction_dir,
                             segmentation, output_template, n_gpus, verbose, threshold, interpolation,
                             segmentation_labels, sum_then_threshold, label_hierarchy, write_input_image=False):
    pred_x = pytorch_predict_batch_array(model, batch, n_gpus=n_gpus)
    for batch_idx in range(len(batch)):
        pred_image = prediction_to_image(pred_x[batch_idx], input_image=batch_references[batch_idx][0],
                                         reference_image=batch_references[batch_idx][1], interpolation=interpolation,
                                         segmentation=segmentation, segmentation_labels=segmentation_labels,
                                         threshold=threshold, sum_then_threshold=sum_then_threshold,
                                         label_hierarchy=label_hierarchy)
        write_prediction_image_to_file(pred_image, output_template,
                                       subject_id=batch_subjects[batch_idx],
                                       x_filename=batch_filenames[batch_idx],
                                       prediction_dir=prediction_dir,
                                       basename=basename,
                                       verbose=verbose)
        if write_input_image:
            write_prediction_image_to_file(batch_references[batch_idx][0], output_template=output_template,
                                           subject_id=batch_subjects[batch_idx] + "_input",
                                           x_filename=batch_filenames[batch_idx],
                                           prediction_dir=prediction_dir,
                                           basename=basename,
                                           verbose=verbose)


def volumetric_predictions(model, dataloader, prediction_dir, activation=None, resample=False,
                           interpolation="trilinear", inferer=None):
    output_filenames = list()
    writer = NibabelWriter()
    if resample:
        resampler = ResampleToMatch(mode=interpolation)
        loader = LoadImage(image_only=True, ensure_channel_first=True)
    print("Dataset: ", len(dataloader))
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            x = item["image"]
            x = x.to(next(model.parameters()).device)  # Set the input to the same device as the model parameters
            if inferer:
                predictions = inferer(x, model)
            else:
                predictions = model(x)
            if activation == "sigmoid":
                predictions = torch.sigmoid(predictions)
            elif activation == "softmax":
                predictions = torch.softmax(predictions, dim=1)
            elif activation is not None:
                predictions = getattr(torch, activation)(predictions)
            batch_size = x.shape[0]
            for batch_idx in range(batch_size):
                _prediction = predictions[batch_idx]
                _x = x[batch_idx]
                if resample:
                    _x = loader(os.path.abspath(_x.meta["filename_or_obj"]))
                    _prediction = resampler(_prediction, _x)
                writer.set_data_array(_prediction)
                writer.set_metadata(_x.meta, resample=False)
                out_filename = os.path.join(prediction_dir,
                                            os.path.basename(_x.meta["filename_or_obj"]).split(".")[0] + ".nii.gz")
                writer.write(out_filename, verbose=True)
                output_filenames.append(out_filename)
    return output_filenames

class FeatureExtractor:
    def __init__(self, model, config_filenames):
        self.model = model
        self.features = []
        self.subject_ids = []
        self.all_subject_ids = self._extract_subject_ids(config_filenames)
        self.handle = None  # 用于存储hook句柄
        
    def _extract_subject_ids(self, config_filenames):
        ids = []
        for item in config_filenames:
            path = item["image"][0]  # 获取第一个image路径
            # 提取ses-后面的数字部分
            ses_part = path.split("/")[-1] 
            ids.append(ses_part)
        return ids
        
    def hook_fn(self, module, input, output):
        pooled = torch.nn.functional.adaptive_avg_pool3d(output, (1, 1, 1))
        flattened = pooled.view(output.size(0), -1)
        self.features.append(flattened.detach().cpu())
        
    def register_hook(self):
        # 确保模型有bottleneck层
        if hasattr(self.model, 'bottleneck') and hasattr(self.model.bottleneck, 'conv2') and hasattr(self.model.bottleneck.conv2, 'conv'):
            target_layer = self.model.bottleneck.conv2.conv
            self.handle = target_layer.register_forward_hook(self.hook_fn)
        else:
            raise AttributeError("Model doesn't have the expected bottleneck.conv2.conv layer")
            
    def remove_hook(self):
        if self.handle is not None:
            self.handle.remove()
        
    def collect_features(self, dataloader):
        self.register_hook()  # 现在这个方法存在了
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x = batch["image"].to(next(self.model.parameters()).device)
                _ = self.model(x)
                
                # 根据batch索引获取对应的subject IDs
                start_idx = batch_idx * x.size(0)
                end_idx = start_idx + x.size(0)
                self.subject_ids.extend(self.all_subject_ids[start_idx:end_idx])
                
        self.remove_hook()
        return torch.cat(self.features, dim=0).numpy(), self.subject_ids
        
def extract_numeric_id(subject_id):
    # 使用正则表达式找到所有连续的数字部分
    numeric_parts = re.findall(r'\d+', subject_id)
    # 用"-"连接所有数字部分
    return '-'.join(numeric_parts) if numeric_parts else ''
    

def create_publication_quality_tsne(features, subject_ids, highlight_ids, output_path):
    # 确保highlight_ids是字符串列表
    highlight_ids = [str(id) for id in highlight_ids]  # 改为直接转为字符串
    
    highlight_labels = []
    for sid in subject_ids:
        # 提取数字ID
        extracted_id = extract_numeric_id(sid)
        # 检查是否在highlight_ids中
        is_highlight = extracted_id in highlight_ids 
        # print(extracted_id)
        highlight_labels.append("Highlighted" if is_highlight else "Others")
    
    # 执行PCA和t-SNE
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(pca_features)
    
    # 计算与highlight点的最远点
    if highlight_ids:
        highlight_indices = [i for i, label in enumerate(highlight_labels) if label == "Highlighted"]
        if highlight_indices:
            # 计算所有点到highlight点的平均距离
            distances = np.mean(np.sqrt(
                (tsne_results[:, 0][:, None] - tsne_results[highlight_indices, 0])**2 +
                (tsne_results[:, 1][:, None] - tsne_results[highlight_indices, 1])**2
            ), axis=1)
            farthest_idx = np.argmax(distances)
            highlight_labels[farthest_idx] = "Farthest"
    
    # 创建DataFrame
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'Subject': subject_ids,
        'Session': [extract_numeric_id(sid) for sid in subject_ids],
        'Group': highlight_labels
    })
    
    # 自定义颜色方案
    color_discrete_map = {
        "Highlighted": "rgb(255, 0, 0)",      # 鲜艳红色
        "Farthest": "rgb(0, 0, 255)",         # 深蓝色
        "Others": "rgb(100, 100, 100)"        # 深灰色
    }
    
    # 创建图表
    fig = px.scatter(
        df, x='x', y='y', color='Group',
        color_discrete_map=color_discrete_map,
        hover_name='Subject',
        hover_data={'x': ':.2f', 'y': ':.2f', 'Session': True, 'Group': False},
        title='t-SNE Visualization',
        width=1000, height=800,
        size_max=10
    )
    
    # 增强对比度和样式
    fig.update_traces(
        marker=dict(
            size=12,
            line=dict(width=1, color='DarkSlateGrey'),
            opacity=0.9
        ),
        selector=dict(mode='markers')
    )
    
    # 高亮点和最远点的特殊样式
    fig.update_traces(
        marker=dict(size=16, line=dict(width=2, color='black')),
        selector=dict(name='Highlighted')
    )
    fig.update_traces(
        marker=dict(size=16, line=dict(width=2, color='black')),
        selector=dict(name='Farthest')
    )
    
    # 图表布局调整
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, family="Arial", color="black"),
        legend_title_text='',
        margin=dict(l=50, r=50, b=50, t=80),
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 坐标轴样式
    fig.update_xaxes(
        title_text='t-SNE Dimension 1',
        showline=True, linewidth=2, linecolor='black',
        gridcolor='lightgray', mirror=True
    )
    fig.update_yaxes(
        title_text='t-SNE Dimension 2',
        showline=True, linewidth=2, linecolor='black',
        gridcolor='lightgray', mirror=True
    )
    
    fig.write_html(output_path)
    print(f"t-SNE plot saved to {output_path}")
    
def volumetric_tsne(model, dataloader, prediction_dir, config_filenames, 
                   activation=None, resample=False, interpolation="trilinear", 
                   inferer=None, highlight_ids=None):
    extractor = FeatureExtractor(model, config_filenames)
    features, subject_ids = extractor.collect_features(dataloader)
    
    os.makedirs(prediction_dir, exist_ok=True)
    output_path = os.path.join(prediction_dir, "tsne_plot.html")
    
    # 确保highlight_ids是列表形式
    if isinstance(highlight_ids, str):
        highlight_ids = eval(highlight_ids) if highlight_ids else []
    elif highlight_ids is None:
        highlight_ids = []
    
    create_publication_quality_tsne(features, subject_ids, highlight_ids, output_path)
    
    return [output_path]
