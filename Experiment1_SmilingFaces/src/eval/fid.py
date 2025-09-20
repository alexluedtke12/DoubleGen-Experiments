import torch, torch_fidelity
# from insightface.model_zoo import model_zoo
# from torch_fidelity.registry import register_feature_extractor

# class ArcFaceExtractor(torch_fidelity.FeatureExtractorBase):
#     def __init__(self, device):
#         super().__init__(device)
#         self.arc = model_zoo.get_model("arcface_r100_v1").to(device).eval()

#     @torch.inference_mode()
#     def forward(self, imgs):                    # imgs: float32 N×3×H×W in [0,1]
#         imgs = torch.nn.functional.interpolate(
#             imgs, (112, 112), mode="bilinear", align_corners=False
#         )
#         return self.arc(imgs).float()           # N×512

# register_feature_extractor("arcface-r100", ArcFaceExtractor)


# for method in ["aipw_wrongProp_wrongOut",
#     "aipw_rightProp_rightOut",
#     "aipw_rightProp_wrongOut",
#     "aipw_wrongProp_rightOut",
#     "ipw_right",
#     "ipw_wrong",
#     "gcomp_right",
#     "gcomp_wrong",
#     "no_wgt"]:
for method in ["aipw_rightProp_rightOut",
    "no_wgt"]:
    real  = "test_cf_images/"
    fake  = f"CelebA-attrs-model/{method}/model_ema/individual_samples/epoch_0499"
    
    # # ArcFace-based FID
    # fid_arc = torch_fidelity.calculate_metrics(
    #     input1=real, input2=fake,
    #     fid=True,
    #     feature_extractor="arcface-r100",
    #     batch_size=128,
    #     cuda=True,
    # )["frechet_inception_distance"]
    
    # Standard Inception-V3 FID
    fid_inc = torch_fidelity.calculate_metrics(
        input1=real, input2=fake,
        fid=True,
        feature_extractor="inception-v3-compat",
        batch_size=128,
        cuda=True,
    )["frechet_inception_distance"]
    
    print(f"Method: {method}; FID (Inception-v3): {fid_inc:.3f}")

