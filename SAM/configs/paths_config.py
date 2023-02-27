# dataset_paths = {
#     'celeba_test': '',
#     'ffhq': '',
# }
dataset_paths = {
    'ffhq': '/path/to/ffhq/images256x256'
    ,'celeba_test': '/path/to/CelebAMask-HQ/test_img',
}

# model_paths = {
#     'pretrained_psp_encoder': 'pretrained_models/psp_ffhq_encode.pt',
#     'ir_se50': 'pretrained_models/model_ir_se50.pth',
#     'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
#     'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
#     'age_predictor': 'pretrained_models/dex_age_classifier.pth'
# }



model_paths = {
    'pretrained_psp': './models/psp.py',
    'pretrained_psp_encoder': './models/psp.py',
    'ir_se50': './models/encoders/model_irse.py',
    'stylegan_ffhq': './models/stylegan2/model.py',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'age_predictor': './models/dex_vgg.py'
}



