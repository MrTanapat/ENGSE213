import matplotlib.pyplot as plt
import numpy as np

years = np.array([2563, 2564, 2565, 2566, 2567])

revenue_apple = np.array([1200, 1350, 1500, 1650, 1800])
revenue_samsung = np.array([950, 1100, 1300, 1450, 1600])
revenue_lg = np.array([700, 900, 1150, 1300, 1500])

y = np.arange(len(years))
height = 0.25

plt.figure(figsize=(10, 8))

plt.barh(y + height, revenue_apple, height, label='Apple',
         color='#ff9999', edgecolor='black')
plt.barh(y, revenue_samsung, height, label='Samsung',
         color='#66b3ff', edgecolor='black')
plt.barh(y - height, revenue_lg, height, label='LG',
         color='#99ff99', edgecolor='black')

plt.title('Revenue Comparison: Apple vs Samsung vs LG (2563-2567)', fontsize=16)
plt.xlabel('Revenue (Million Baht)', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.yticks(y, years)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
