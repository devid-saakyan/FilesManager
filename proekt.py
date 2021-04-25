import os

'''inputs = new
double[784];
for (int i = 0; i < w; i++)
{
for (int j = 0; j < h; j++) {
if (mousePressed != 0) {
double dist = (i - mx) * (i - mx) + (j - my) * (j - my);
if (dist < 1) dist = 1;
dist *= dist;
if (mousePressed == 1) colors[i][j] += 0.1 / dist;
else colors[i][j] -= 0.1 / dist;
if (colors[i][j] > 1) colors[i][j] = 1;
if (colors[i][j] < 0) colors[i][j] = 0;
}
int color = (int)(colors[i][j] * 255);
color = (color << 16) | (color << 8) | color;
pimg.setRGB(i, j, color);
inputs[i + j * w] = colors[i][j];'''

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation

def object_detection_on_an_image():
    segment_image = instance_segmentation()
    segment_image.load_model(r"C:\Users\AFK\PycharmProjects\pythonProject\proekt\mask_rcnn_coco.h5")

    target_class = segment_image.select_target_classes(person=True)

    result = segment_image.segmentImage(
        # image_path="1city.jpg",
        # image_path="2cars_people.jpeg",          }
'''                double[] outputs = nn.feedForward(inputs);
                int maxDigit = 0;
                double maxDigitWeight = -1;
                for (int i = 0; i < 10; i++) {
                    if(outputs[i] > maxDigitWeight) {
                        maxDigitWeight = outputs[i];
                        maxDigit = i;
                    }
                }
                Graphics2D ig = (Graphics2D) img.getGraphics();
                ig.drawImage(pimg, 0, 0, w * scale, h * scale, this);
                ig.setColor(Color.lightGray);
                ig.fillRect(w * scale + 1, 0, 200, h * scale);
                ig.setFont(new Font("TimesRoman", Font.BOLD, 48));
                for (int i = 0; i < 10; i++) {
                    if(maxDigit == i) ig.setColor(Color.RED);
                    else ig.setColor(Color.GRAY);
                    ig.drawString(i + ":", w * scale + 20, i * w * scale / 15 + 150);
                    Color rectColor = new Color(0, (float)outputs[i], 0);
                    int rectWidth = (int)(outputs[i] * 100);
                    ig.setColor(rectColor);
                    ig.fillRect(w * scale + 70, i * w * scale / 15 + 122, rectWidth, 30);
                }
                g.drawImage(img, 8, 30, w * scale + 200, h * scale, this);
                frame++;
            }            '''
        image_path="3silicon_valley.jpg",
        # show_bboxes=True,
        segment_target_classes=target_class,
        # extract_segmented_objects=True,
        # save_extracted_objects=True,
        output_image_name="output.jpg"
    )

    print(result[0]["scores"])
    objects_count = len(result[0]["scores"])
    print(f"Найдено объектов: {objects_count}")

def main():
    object_detection_on_an_image()

if __name__ == '__main__':
    main()