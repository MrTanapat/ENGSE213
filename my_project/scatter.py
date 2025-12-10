import matplotlib.pyplot as plt
import numpy as np

years = np.array([2563, 2564, 2565, 2566, 2567])

revenue_apple = np.array([1200, 1350, 1500, 1650, 1800])
revenue_samsung = np.array([950, 1100, 1300, 1450, 1600])
revenue_lg = np.array([700, 900, 1150, 1300, 1500])

plt.figure(figsize=(10, 6))

plt.scatter(years, revenue_apple, color='#ff9999',
            label='Apple', s=100, marker='o', edgecolors='black')
plt.scatter(years, revenue_samsung, color='#66b3ff',
            label='Samsung', s=100, marker='s', edgecolors='black')
plt.scatter(years, revenue_lg, color='#99ff99', label='LG',
            s=100, marker='^', edgecolors='black')

plt.title('Revenue Comparison: Apple vs Samsung vs LG (2563-2567)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (Million Baht)', fontsize=12)

plt.xticks(years)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
