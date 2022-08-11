import cv2
import numpy as np


def main(file):    
    # 画像白黒で読み込み
    image = cv2.imread(file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 画像サイズによりカーネル決定
    character_vertical_kernel_hight = int(image.shape[0] / 130)     # 縦
    character_vertical_kernel_width = int(image.shape[1] / 130)     # 横
    character_vertical_iterations = 2                               # 誇張処理回数

    # 2値化
    threshold = 100.0
    maxvalue = 255.0
    _, binary_image = cv2.threshold(gray_image, threshold, maxvalue, cv2.THRESH_BINARY)
    # 白黒反転
    binary_image = 255 - binary_image

    # 白部分の膨張処理（Dilation）：モルフォロジー変換 - 2値画像を対象
    kernel = np.ones((character_vertical_kernel_hight, character_vertical_kernel_width),  np.uint8)
    binary_image_dilation = cv2.dilate(binary_image, kernel, iterations = character_vertical_iterations)
    # cv2.imwrite('tmp.png', binary_image_dilation)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(binary_image_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        boxes += [[x, y, x + w, y + h]]
    
    checked_boxes = ProcessingBox.check_include_box(boxes)
    checked_boxes = ProcessingBox.check_cross_box(boxes)

    # 輪郭を描画
    for box in checked_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv2.imwrite('result.png', image)


class ProcessingBox:

    @classmethod
    def check_include_box(cls, boxes):
        include_boxes = []
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                # 片方のboxが片方のboxを内包しているかチェックする
                x01, y01, x02, y02 = boxes[i]
                x11, y11, x12, y12 = boxes[j]
                if x01 < x11 and y01 < y11 and x02 > x12 and y02 > y12:
                    include_boxes += [boxes[j]]
        # 内包しているboxを削除
        for box in include_boxes:
            boxes.remove(box)

        return boxes

    @classmethod
    def check_cross_box(cls, boxes):
        "交差座標を結合する"
        calc_boxes = []
        exclude_box = []
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                tuple_box_1 = cls.__calc_vertical_horizontal(boxes[i])
                tuple_box_2 = cls.__calc_vertical_horizontal(boxes[j])
                retval, intersecting_region = cv2.rotatedRectangleIntersection(tuple_box_1, tuple_box_2)
                # box同士の交差なし
                if retval == 0:
                    calc_boxes += [boxes[i]]
                # box同士が交差していたら
                if retval == 1:
                    union_box = cls.__union_box(boxes[i], boxes[j])
                    calc_boxes += [union_box]
                    exclude_box += [boxes[i]] + [boxes[j]]
        # 重複座標を削除
        calc_boxes = list(set(map(tuple, calc_boxes)))
        exclude_box = list(set(map(tuple, exclude_box)))
        # 交差座標を削除
        for box in exclude_box:
            calc_boxes.remove(box)
        return calc_boxes
            
    @classmethod
    def __union_box(cls, box1, box2):
        "2つの座標リストを結合する"
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2

        minx = min([x01, x02, x11, x12])
        maxx = max([x01, x02, x11, x12])
        miny = min([y01, y02, y11, y12])
        maxy = max([y01, y02, y11, y12])

        union_box = [minx, miny, maxx, maxy]
        return union_box

    @classmethod
    def __calc_vertical_horizontal(cls, box):
        "左上＆右下の座標を受け取り、真ん中座標＆幅＆高さに変形する"
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return ((x_center, y_center), (w, h), 0)


if __name__ == '__main__':
    file = "sample.png"
    main(file)