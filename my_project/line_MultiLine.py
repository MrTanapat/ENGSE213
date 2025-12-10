import matplotlib.pyplot as plt
import numpy as np

years = np.array([2563, 2564, 2565, 2566, 2567])

revenue_apple = np.array([1200, 1350, 1500, 1650, 1800])
revenue_samsung = np.array([950, 1100, 1300, 1450, 1600])
revenue_lg = np.array([700, 900, 1150, 1300, 1500])

plt.figure(figsize=(10, 6))

plt.plot(years, revenue_apple,
         marker='o', linestyle='-', color='#ff9999', linewidth=2, label='Apple')
plt.plot(years, revenue_samsung,
         marker='s', linestyle='--', color='#66b3ff', linewidth=2, label='Samsung')
plt.plot(years, revenue_lg,
         marker='^', linestyle='-.', color='#99ff99', linewidth=2, label='LG')

plt.title('Revenue Comparison: Apple vs Samsung vs LG (2563-2567)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (Million Baht)', fontsize=12)
plt.xticks(years)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
