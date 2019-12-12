from numpy import loadtxt
from keras.models import load_model
from LoadData import LoadOneBWTestData, LoadOneColorTestData
from metrics import precision_m
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def Nmaxelements(list1, N):
    final_list = []
    list1 = list1.copy()

    for i in range(0, N):
        max1 = max(list1)
        list1.remove(max1);
        final_list.append(max1)
    return final_list

def get_highest_waldo_vals(predictions, size_tuple):
    locations = []
    predictions = list(map(lambda p: p[1], predictions.tolist()))
    top_predictions = Nmaxelements(predictions, 3)
    print(top_predictions)
    for prediction in top_predictions:
        if(len(locations) == 3):
            break
        for i, x in enumerate(predictions):
            if(len(locations) == 3):
                break
            if(x == prediction):
                locations.append(square_location(i, 64, size_tuple))
    return locations


def square_location(img_order, chop_size, size_tuple):
    rangeX = int(size_tuple[0] / chop_size)
    rangeY = int(size_tuple[1] / chop_size)
    x_location = int(img_order/rangeY)
    y_location = img_order%rangeY

    return (x_location*chop_size, y_location*chop_size)

def calc_square_locations(predictions, size_tuple):
    locations = []
    for i in range(len(predictions)):
      if(int(predictions[i][0]) == 1):
          locations.append(square_location(i, 64, size_tuple))
    return locations


test_images, original_image, img_size = LoadOneColorTestData('original-images/9.jpg', 64)


model = load_model('model-64-expanded.h5')

predictions = model.predict(test_images)

# print(predictions)

with open("predictions.csv","w") as pred_file:
    pred_file.write("Index,IsWaldo\n")
    for i in range(len(test_images)):
        pred_file.write(str(i)+","+str(predictions[i][1])+"\n")

locations = get_highest_waldo_vals(predictions, img_size)

fig, ax = plt.subplots()
ax.imshow(original_image)
for location in locations:
    rect = patches.Rectangle(location,64,64,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()
