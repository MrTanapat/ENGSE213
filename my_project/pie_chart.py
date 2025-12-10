import matplotlib.pyplot as plt
import numpy as np

years = np.array([2563, 2564, 2565, 2566, 2567])

revenue_apple = np.array([1200, 1350, 1500, 1650, 1800])
revenue_samsung = np.array([950, 1100, 1300, 1450, 1600])
revenue_lg = np.array([700, 900, 1150, 1300, 1500])

total_apple = np.sum(revenue_apple)
total_samsung = np.sum(revenue_samsung)
total_lg = np.sum(revenue_lg)

labels = ['Apple', 'Samsung', 'LG']
sizes = [total_apple, total_samsung, total_lg]
colors = ['#ff9999', '#66b3ff', '#99ff99']

plt.figure(figsize=(8, 8))
plt.pie(sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90)

plt.title('Revenue Comparison: Apple vs Samsung vs LG (2563-2567)', fontsize=16)
plt.show()
