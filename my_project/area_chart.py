import matplotlib.pyplot as plt
import numpy as np

years = np.array([2563, 2564, 2565, 2566, 2567])

revenue_apple = np.array([1200, 1350, 1500, 1650, 1800])
revenue_samsung = np.array([950, 1100, 1300, 1450, 1600])
revenue_lg = np.array([700, 900, 1150, 1300, 1500])

plt.figure(figsize=(10, 6))

plt.stackplot(years, revenue_apple, revenue_samsung, revenue_lg,
              labels=['Apple', 'Samsung', 'LG'],
              colors=['#ff9999', '#66b3ff', '#99ff99'],
              alpha=0.8)

y1 = revenue_apple
y2 = revenue_apple + revenue_samsung
y3 = revenue_apple + revenue_samsung + revenue_lg

plt.plot(years, y1, color='#cc0000', linewidth=2,
         marker='o', label='_nolegend_')
plt.plot(years, y2, color='#0055cc', linewidth=2,
         marker='o', label='_nolegend_')
plt.plot(years, y3, color='#009900', linewidth=2,
         marker='o', label='_nolegend_')

plt.title('Revenue Comparison: Apple vs Samsung vs LG (2563-2567)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Revenue (Million Baht)', fontsize=12)
plt.xticks(years)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
