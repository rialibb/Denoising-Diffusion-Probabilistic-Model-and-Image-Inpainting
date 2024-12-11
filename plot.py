from schedules import linear_schedule, cosine_schedule, quadratic_schedule, exponential_schedule, logarithmic_schedule
import matplotlib.pyplot as plt
import numpy as np






k = np.arange(1000)

a1 = linear_schedule()
a2 = cosine_schedule()
a3 = quadratic_schedule()
a4 = exponential_schedule()
a5 = logarithmic_schedule()

plt.plot(k, a1[k], label='Linear Schedule')
plt.plot(k, a2[k], label='Cosine Schedule')
plt.plot(k, a3[k], label='Quadratic Schedule')
plt.plot(k, a4[k], label='Exponential Schedule')
plt.plot(k, a5[k], label='Logarithmic Schedule')
plt.title('shape of Schedule')
plt.xlabel('timestep')
plt.ylabel('value')


plt.legend()
plt.savefig('Schedules_plot.png', dpi=300)
plt.show()

# I tested the validation mse for these 5 schedules, I found that the best validation is ordered in the same way as the curve (val_mse_log < val_mse_linear < quadratic < expo < <cosine )
# --> we can't say anything rigt now, as the mse concentrate on local error(pixels), we need also to see the global distribution error FID --> will implement it in image inpainting