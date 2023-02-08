# model_path = r"/home/turnerp/PycharmProjects/GAPSEQML/models/TEST_221220_1745/inceptiontime_model"
# state_dict = torch.load(model_path)["model_state_dict"]

# traces, label = next(iter(testloader))
# model.load_state_dict(state_dict)

# # with open('gradcam.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
# #     pickle.dump([model, traces], f)

# with open('gradcam.pkl','rb') as f:  # Python 3: open(..., 'rb')
#     model, traces = pickle.load(f)

# traces = traces.to(device)

# class GradCAM():

#     def __init__(self, model, target_layer):

#         self.model = model
#         self.target_layer = target_layer

#         self.gradients = dict()
#         self.activations = dict()


#         def backward_hook(module, grad_input, grad_output):
#           self.gradients['value'] = grad_output[0]

#         def forward_hook(module, input, output):
#             self.activations['value'] = output

#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_backward_hook(backward_hook)

#     def forward(self, input, class_idx=None, retain_graph=False):
#         b, c, l = input.size()

#         # self.model.eval()

#         logit = self.model(input)

#         if class_idx is None:
#             score = logit[:, logit.max(1)[-1]].squeeze()
#         else:
#             score = logit[:, class_idx].squeeze()

#         self.model.zero_grad()
#         score.backward(retain_graph=retain_graph)
#         # gradients = self.gradients['value']
#         # activations = self.activations['value']
#         # b, k, u, v = gradients.size()


#         return score


#     def __call__(self, input):
#         return self.forward(input, class_idx=0, retain_graph=False)


# target_layer = model.inceptionblock.inception[5]

# gradcam = GradCAM(model, target_layer)

# output = gradcam(traces)

# output = output.cpu().detach().numpy()


# target_layers = [model.inceptionblock.inception[5]]

# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# targets = [ClassifierOutputTarget(0)]


# traces, label = next(iter(testloader))

# traces, label = traces.to(device), label.to(device)

# grayscale_cam = cam(input_tensor=traces)


# import shap

# train_images = torch.from_numpy(np.stack(X[:100])).float().unsqueeze(1)
# deep_explainer = shap.DeepExplainer(model.eval(), train_images)

# shap_values = deep_explainer.shap_values(traces)

# x1 = shap_values[0][0][0]
# x2 = shap_values[1][0][0]

# plt.plot(traces.numpy()[0][0])
# plt.show()
# plt.plot(x1)
# plt.show()
# plt.plot(x2)
# plt.plot()


#     saliency_maps.append(shap_img)


# model.eval()


# #we want to calculate gradient of higest score w.r.t. input
# #so set requires_grad to True for input
# traces.requires_grad = True
# #forward pass to calculate predictions
# preds = model(traces)
# score, indices = torch.max(preds, 1)
# # #backward pass to get gradients of score predicted class w.r.t. input image
# score.backward()
# # #get max along channel axis
# slc, _ = torch.max(torch.abs(traces.grad[0]), dim=0)
# # #normalize to [0..1]
# slc = (slc - slc.min())/(slc.max()-slc.min())
# slc = slc.cpu().numpy()

# traces = traces.data.cpu().numpy()[0][0]

# plt.plot(slc)
# plt.show()
# plt.plot(traces)
# plt.show()


# model.load_state_dict(state_dict)
# model.eval()

# state_dict_save = torch.load(model_path)["model_state_dict"]
# model.load_state_dict(state_dict_save)
# model.eval()


# model_path = r"/home/turnerp/PycharmProjects/GAPSEQML/models/TEST_221220_1307/inceptiontime_model"
# model_data = trainer.evaluate(testloader,model_path)


# model_data = trainer.evaluate(testloader,model_path)

# # show = True

# model_data = torch.load(model_path)

# model = InceptionTime(1,len(np.unique(y)))
# model = model.load_state_dict(model_data['model_state_dict'])












