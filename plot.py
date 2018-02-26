import matplotlib.pyplot as plt

fig, ax = plt.subplots()

b = [4, 8, 12, 16, 24, 32, 48]

br = [0.636510551, 0.850447237, 0.935318768, 0.936076641, 0.939434171, 0.935359955, 0.927146196]

ar = [0.657169223, 0.851807237, 0.937406659, 0.93762511, 0.939580858, 0.935836911, 0.931887388]

br2 = [0.428904355, 0.693660557, 0.746403217, 0.759822547, 0.770629287, 0.78904283, 0.791882753]

ar2 = [0.441049755, 0.707811177, 0.758414209, 0.770174026, 0.777347207, 0.791737854, 0.791944087]

plt.plot(b, br, label="CIFAR10 Full. No rotation")
plt.plot(b, ar, label="CIFAR10 Full. With rotation")

plt.plot(b, br2, label="CIFAR10 Reduced. No rotation")
plt.plot(b, ar2, label="CIFAR10 Reduced. With rotation")

plt.legend(loc="lower right", bbox_to_anchor=[1, 0],
           ncol=1)
ax.get_legend().get_title().set_color("red")

ax.tick_params(width=1)

plt.ylabel('mAP')
plt.xlabel('Number of bits')

plt.savefig('rotation.eps')
plt.savefig('rotation.pdf')

plt.show()

