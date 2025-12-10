import matplotlib.pyplot as plt
import numpy as np

years = np.array([2563, 2564, 2565, 2566, 2567])

revenue_apple = np.array([1200, 1350, 1500, 1650, 1800])
revenue_sumsung = np.array([950, 1100, 1300, 1450, 1600])
revenue_lg = np.array([700, 900, 1150, 1300, 1500])

plt.figure(figsize=(10, 6))
plt.plot(years, revenue_apple, marker='o', label='Apple', color='red')
plt.plot(years, revenue_sumsung, marker='s', label='Samsung', color='blue')
plt.plot(years, revenue_lg, marker='^', label='LG', color='green')

plt.title('Revenue Comparison: Apple vs Samsung vs LG (2563-2567)')
plt.xlabel('Year')
plt.ylabel('Revenue (Million Baht)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(years)
plt.legend()
plt.tight_layout()

plt.show()
