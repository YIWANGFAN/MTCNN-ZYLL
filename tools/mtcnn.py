import cv2
from tools.detect import create_mtcnn_net, MtcnnDetector






def demo(path):

    #trained model
    p_model_path = "./model/pnet_epoch_train.pt"
    r_model_path = "./model/rnet_epoch_train.pt"
    o_model_path = "./model/onet_epoch_train.pt"
    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path, use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24,threshold=[0.6, 0.7, 0.7])

    img = cv2.imread(path)

    bboxs = mtcnn_detector.detect_face(img)
    # print box_align

    c = 0
    st = 0
    for i in range(bboxs.shape[0]):
        b = bboxs[i, :4]
        l = b[2] - b[0]
        h = b[3] - b[1]
        s = l * h
        if st < s:
            st = s
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)

            image_r = img[int(b[1]) - 10:  int(b[3]) + 10, int(b[0]) - 10: int(b[2]) + 10]

            # cv2.imwrite("./img/" + "wxf" + ".jpg", image_r, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


    image = cv2.cvtColor(image_r, cv2.COLOR_BGR2HSV)
    sumRed = 0
    sumYello = 0
    sumblack = 0
    sumwhite = 0
    sumcyan = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j, 0] <= 180 & image[i, j, 0] >= 156) & (image[i, j, 1] >= 43
                                                                  & image[i, j, 1] <= 255) & (
                    image[i, j, 2] >= 46 & image[i, j, 2] <= 255):
                sumRed = sumRed + 1
            if (image[i, j, 0] <= 10 & image[i, j, 0] >= 0) & (image[i, j, 1] >= 43
                                                               & image[i, j, 1] <= 255) & (
                    image[i, j, 2] >= 46 & image[i, j, 2] <= 255):
                sumRed = sumRed + 1
            if (image[i, j, 0] <= 180 & image[i, j, 0] >= 0) & (image[i, j, 1] >= 0
                                                                & image[i, j, 1] <= 255) & (
                    image[i, j, 2] >= 0 & image[i, j, 2] <= 46):
                sumblack = sumblack + 1
            if (image[i, j, 0] <= 180 & image[i, j, 0] >= 0) & (image[i, j, 1] >= 0
                                                                & image[i, j, 1] <= 30) & (
                    image[i, j, 2] >= 221 & image[i, j, 2] <= 255):
                sumwhite = sumwhite + 1
            if (image[i, j, 0] <= 255 & image[i, j, 0] >= 34) & (image[i, j, 1] >= 43
                                                                 & image[i, j, 1] <= 255) & (
                    image[i, j, 2] >= 46 & image[i, j, 2] <= 255):
                sumYello = sumYello + 1
            if (image[i, j, 0] <= 99 & image[i, j, 0] >= 78) & (image[i, j, 1] >= 43
                                                                & image[i, j, 1] <= 255) & (
                    image[i, j, 2] >= 46 & image[i, j, 2] <= 255):
                sumcyan = sumcyan + 1
    all = image.shape[0] * image.shape[1]
    red = sumRed / all
    yello = sumYello / all
    black = sumblack / all
    cyan = sumcyan / all
    white = sumwhite / all
    print('red:', red, 'yello:', yello, 'white:', white, 'black:', black, 'cyan:', cyan)