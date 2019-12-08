import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
import json
import os

MIN_MATCH_COUNT = 10


def compare_images(image1_path, image2_path):
    im1 = cv.imread(image1_path, 0)
    im2 = cv.imread(image2_path, 0)

    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    #
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good_matches.append(m)

    images_match = False
    if len(good_matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        if len([x for x in matchesMask if x == 1])>10:
            images_match = True

        # h,w = im1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv.perspectiveTransform(pts,M)
        #
        # img2 = cv.polylines(im2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        # print("Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT))
        matchesMask = None
        return False

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)
    #
    # img3 = cv.drawMatches(im1,kp1,img2,kp2,good_matches,None,**draw_params)
    #
    # plt.imsave('plot.png', img3)
    return images_match
    # plt.imshow(img3, 'gray')
    # plt.show()

if __name__ == "__main__":
    #compare_images('./campaign_images/485930209/890b12a110f291a703756f1575367e38_original.png', './videos/4/000005.jpg')
    with open('videos/mappings.json','r') as video_mappings_file:
        video_mappings = json.load(video_mappings_file)
    videos_list = [x.split('.mp4')[0] for x in video_mappings.keys()]

    num_videos = len(videos_list)
    video_ci_mappings = {}
    for i, video_id in enumerate(videos_list):
        mapping = video_mappings[video_id+'.mp4']
        frames = os.listdir('./videos/%d/' % mapping )
        frames = [os.path.join('./videos/'+str(mapping), x) for x in frames]
        campaign_images = os.listdir('./campaign_images/%s/' % video_id)
        campaign_images = [os.path.join('./campaign_images/%s/' % video_id, x) for x in campaign_images]
        matching_images = []
        for c_img in campaign_images:
            for frame in frames:
                try:
                    if compare_images(c_img, frame):
                        matching_images.append(c_img)
                        break
                except Exception as e:
                    print(e)

        video_ci_mappings[video_id] = matching_images
        print('%d/%d processed' % (i, num_videos))

    with open('campaign_images_used.json', 'w') as ci_file:
        json.dump(video_ci_mappings, ci_file)

