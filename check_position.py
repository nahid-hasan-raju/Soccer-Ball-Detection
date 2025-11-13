import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv('Soccerball/test/_annotations.csv')

# Show what classes exist
print("Classes in CSV:")
print(df['class'].value_counts())
print()

# Filter for ball
ball_df = df[df['class'].str.contains('ball', case=False, na=False)]
print(f"Found {len(ball_df)} ball annotations")
print()

# Check FIRST image
row = ball_df.iloc[15]
print(f"First image: {row['filename']}")
print(f"Class: {row['class']}")
print(f"BBox: ({row['xmin']}, {row['ymin']}) to ({row['xmax']}, {row['ymax']})")
print()

# Load and visualize
img_path = os.path.join('Soccerball/test', row['filename'])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Draw the bounding box
cv2.rectangle(img, 
             (int(row['xmin']), int(row['ymin'])),
             (int(row['xmax']), int(row['ymax'])),
             (0, 255, 0), 5)  # GREEN box, thick line

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.title(f'Is the GREEN box on the ball?\n{row["filename"]}')
plt.axis('off')
plt.savefig('CHECK_THIS.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Saved CHECK_THIS.png")
print("LOOK AT IT - Is the green box on the ball?")