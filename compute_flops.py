import torch
from thop import profile
from e2emef import E2EMEF
# from args_fusion import args
# from net import New_net

# in_c = 1
# if in_c == 1:
#     out_c = in_c
#     mode = 'L'
#     model_path = args.model_path_gray
# else:
#     out_c = in_c
#     mode = 'RGB'
#     model_path = args.model_path_rgb


model = E2EMEF(is_guided=True)#.cuda()
# nest_model.load_state_dict(torch.load(model_path))
# nest_model.encoder()
# nest_model.fusion()
# nest_model.decoder()
# nest_model.load_state_dict(torch.load(model_path))


#net = Model()  # 定义好的网络模型
inputa = torch.randn(1, 1, 256, 256)#.cuda()
inputb = torch.randn(1, 1, 256, 256)#.cuda()

# en_r = nest_model.encoder(inputa)
# en_v = nest_model.encoder(inputb)
# f = nest_model.fusion(en_r, en_v, strategy_type='AVG')
# img_fusion = nest_model.decoder(f);

# encoder = nest_model.encoder(inputa)
# fusion = nest_model.fusion(inputb)
# decoder = nest_model.decoder(inputa,inputb)

Macs, params = profile(model, (inputa,inputb))
print('flops: %.2f ' % (Macs * 2))
print('flops: %.2fG ' % ((Macs * 2) / 1e9), 'params: %.3fM '% (params / 1e6))
# print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
# print('  + Number of params: %.2fM' % (total / 1e6))

# flops_1, params_1 = profile(nest_model.encoder, (inputa,))
# flops_2, params_2 = profile(nest_model.fusion, (en_r,en_v,))
# flops_3, params_3 = profile(nest_model.decoder, (f,))
# print('flops_1: ', flops_1, 'params_1: ', params_1)
# print('flops_2: ', flops_2, 'params_2: ', params_2)
# print('flops_3: ', flops_3, 'params_3: ', params_3)

