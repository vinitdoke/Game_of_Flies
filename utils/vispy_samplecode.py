import vispy
print(vispy.sys_info())

# import numpy as np
# import vispy
# import vispy.scene
# from vispy.scene import visuals
# from vispy import app
#
# canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
# view = canvas.central_widget.add_view()
# view.camera = 'turntable'
#
#
# # generate data
# def solver(t):
#     # pos = np.array([[0.5 + t/10000, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]])
#     pos = np.random.uniform(0, 100, (100, 3))
#     return pos
#
#
# # These are the data that need to be updated each frame --^
#
# scatter = visuals.Markers()
# view.add(scatter)
#
# # view.camera = scene.TurntableCamera(up='z')
#
# # just makes the axes
# axis = visuals.XYZAxis(parent=view.scene)
#
# t = 0.0
#
#
# def update(ev):
#     global scatter
#     global t
#     t += 1.0
#     scatter.set_data(solver(t), edge_color=None, face_color=(1, 1, 1, .5),
#                      size=50)
#
#
# timer = app.Timer()
# timer.connect(update)
# timer.start(0)
# if __name__ == '__main__':
#     canvas.show()
#     if 1:
#         app.run()
#
# # import numpy as np
# #
# # import vispy.plot as vp
# #
# # np.random.seed(2324)
# # n = 100000
# # data = np.empty((n, 2))
# # lasti = 0
# # for i in range(1, 20):
# #     nexti = lasti + (n - lasti) // 2
# #     scale = np.abs(np.random.randn(2)) + 0.1
# #     scale[1] = scale.mean()
# #     data[lasti:nexti] = np.random.normal(size=(nexti-lasti, 2),
# #                                          loc=np.random.randn(2),
# #                                          scale=scale / i)
# #     lasti = nexti
# # data = data[:lasti]
# #
# #
# # color = (1,0,0)
# # n_bins = 100
# #
# # fig = vp.Fig(show=False)
# # line = fig[0:4, 0:4].plot(data, symbol='o', width=0,
# #                           face_color=color + (0.02,), edge_color=None,
# #                           marker_size=20)
# # line.set_gl_state(depth_test=False)
# # fig[4, 0:4].histogram(data[:, 0], bins=n_bins, color=color, orientation='h')
# # fig[0:4, 4].histogram(data[:, 1], bins=n_bins, color=color, orientation='v')
# #
# # if __name__ == '__main__':
# #     fig.show(run=True)
