import cv2 
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
def drawKeypoints(vis, keypoints, color = (100, 120, 255)):
    for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), int(kp.size), color)
    return vis
def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T
def compute_pose_from_essential(E):
    U,S,V = np.linalg.svd(E) 
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    H = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
        np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T ,
        np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
        np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]   
    print("translation",U[:,2])

    return H    
#reading image
def extrinsic_from_camera_pose(m_c1_wrt_c2):
    H_m = np.vstack([m_c1_wrt_c2, [0, 0, 0, 1]])
    ext = np.linalg.inv(H_m)
    return ext
img1 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/inew1.jpg') 
img1 = cv2.resize(img1, (0,0), fx=0.2, fy=0.2) 
def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    return np.array([
        [0, -1, x[1]],
        [1, 0, -x[0]],
        [-x[1], x[0], 0]
    ])
def linear_triangulation(p1, p2, m1, m2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res    
def reconstruct_one_point(pt1,pt2,m1,m2):
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def cart2hom(arr):
    """ Convert catesian to homogenous points by appending a row of 1s
    :param arr: array of shape (num_dimension x num_points)
    :returns: array of shape ((num_dimension+1) x num_points) 
    """
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

def find_corresponding(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

    # img1 = drawKeypoints(gray1,keypoints_1)
    # cv2.imshow('img1',img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    # img2 = drawKeypoints(gray2,keypoints_2)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    src_pts = np.asarray([keypoints_1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([keypoints_2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = None,
                   flags = 0)
    # img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,matches,None, matchColor=(0, 255, 0), matchesMask=None,singlePointColor=(255, 0, 0), flags=0)       
    # cv2.imshow('img3',img3) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()       
    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]
    # print(pts1)
    return pts1.T,pts2.T

def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    """ Compute the fundamental or essential matrix from corresponding points
        (x1, x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def scale_and_translate_points(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: array of homogenous point (3 x n)
    :returns: array of same input shape and its normalization matrix
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ Computes the fundamental or essential matrix from corresponding points
        using the normalized 8 point algorithm.
    :input p1, p2: corresponding points with shape 3 x n
    :returns: fundamental or essential matrix with shape 3 x 3
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]
def drawlines(img1, img2, lines, pts1, pts2): 
    
    r, c = img1.shape 
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
    cnt=0  
    for r, pt1, pt2 in zip(lines, pts1, pts2): 
          
        color = tuple(np.random.randint(0, 255, 
                                        3).tolist()) 
          
        x0, y0 = map(int, [0, -r[2] / r[1] ]) 
        x1, y1 = map(int,  
                     [c, -(r[2] + r[0] * c) / r[1] ]) 
          
        img1 = cv2.line(img1,  
                        (x0, y0), (x1, y1), color, 1) 
        img1 = cv2.circle(img1, 
                          tuple(pt1), 5, color, -1) 
        img2 = cv2.circle(img2,  
                          tuple(pt2), 5, color, -1) 
        cv2.imshow("img1",img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()     
        if(cnt==10):
            break     
        cnt+=1                
    return img1, img2 
def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)

def nlTriangulation(p1,p2,F):
    J = np.vstack((np.dot(F.T,p1)[0],np.dot(F.T,p1)[1]))
    J = np.vstack((J,np.vstack((np.dot(F,p2)[0],np.dot(F,p2)[1]))))
    s = (np.dot(F,p2)[0])**2+(np.dot(F,p2)[1])**2 + (np.dot(F.T,p1)[0])**2 + (np.dot(F.T,p1)[1]**2)
    delta = np.dot(p1.T,np.dot(F,p2))*J/s
    x_new = np.vstack((np.vstack((p2[0],p2[1])),np.vstack((p1[0],p1[1])))) - delta
    return x_new
with open("cam_params.txt","w") as f1:
    for j in range(1):
        img1 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/images/dino/img{}.ppm'.format(1))   
        # img1 = cv2.resize(img1, (0,0), fx=.2, fy=.15) 
        # print(img1.shape)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/images/dino/img{}.ppm'.format(j+2))
        # img2 = cv2.resize(img2, (0,0), fx=.2, fy=.15) 

        # print(img2.shape)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #keypoints
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

        img_1 = drawKeypoints(gray1,keypoints_1)
        # cv2.imshow('/img1',img_1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #keypoints
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

        img_2 = drawKeypoints(gray2,keypoints_2)
        # cv2.imshow('img2',img_2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        p1,p2 = find_corresponding(img1,img2)
        p1 = np.float32(p1)
        p2 = np.float32(p2)
        N =  p1.shape[1]
        h,w,ch = img1.shape
        # print(p1.shape)
        
        # E = 
        # mydata
        # K = np.array([[739.07654683  , 0.     ,    w/2],
        # [  0.     ,    735.13429576 ,h/2 ],
        # [  0.      ,     0. ,          1.        ]])

        # try different K and image captured from camera with that K  
        #crazyhorse
        # K = np.array([[2500, 0, w/2],[0, 2500, h/2],[0, 0, 1]])
        # dino
        K = np.array([[2360  , 0.     ,   w/2],
         [  0.    ,2360 ,h/2 ],
         [  0.      ,     0. ,          1.        ]])
        
        # dinoring
        # K = np.array([[3310.400000, 0.000000, 316.730000],[ 0.000000, 3325.500000, 200.550000] ,[0.000000,0.000000, 1.000000]])
        
        #viking
        # K = np.array([[523.81, 0.00 ,252.00],[ 0.00, 523.81, 336.00],[ 0.00, 0.00, 1.00]])
        
        #templering
        # K = np.array([[1520.400000, 0.0, 302.320000] ,[0.000000, 1525.900000, 246.870000],[ 0.000000, 0.000000 ,1.000000 ]] )
        
        # p1_10 = p1[:,0:10]
        
        # p2_10 = p2[:,0:10]
        # F,mask = cv2.findFundamentalMat(p1.T, p2.T, cv2.FM_RANSAC,3, 0.99)
        E,mask = cv2.findEssentialMat(p1.T, p2.T, K, cv2.FM_RANSAC, 0.999, 1.0)
        print("E",E)
    # p2.T
        p1 = p1.T
        p2 = p2.T
        p1 = p1[mask.ravel() == 1] 
        p2 = p2[mask.ravel() == 1] 
        N =  p1.shape[0]
        p1 = p1.T
        p2 = p2.T
        p1n = np.dot(np.linalg.inv(K),np.vstack((p1,np.ones((1,N)))))
        p2n = np.dot(np.linalg.inv(K),np.vstack((p2,np.ones((1,N)))))
        # print("N",N)
        # p1
        # p1 = p1.T
        # p2 = p2.T
        # E = compute_essential_normalized(p1n, p2n)
        F = np.dot(np.linalg.inv(K.T),np.dot(E,np.linalg.inv(K)))
        # print("E",E)
        img1copy = img1.copy()
        img2copy = img2.copy()
        p1x = skew(p1n[0])
        p2x = skew(p2n[0])


       
        for i in range(N):   
            # % The product l=E*p2 is the equation of the epipolar line corresponding
            # % to p2, in the first image.  Here, l=(a,b,c), and the equation of the
            # % line is ax + by + c = 0.
            p2i = np.concatenate((np.array(p2)[:,i],[1]))
            p1i = np.concatenate((np.array(p1)[:,i],[1]))
            l = np.dot(F.T, p2i.T)
            # print("p1=",p1i,"\n")
            # % Calculate residual error.  The product p1'*E*p2 should = 0.  The
            # % difference is the residual.
            res = np.dot(np.dot(p1i.T, F.T) , p2i)
            if(res > 0.1):
                print("res = ",res,"\n")
            start_point = (600,int((-l[2]-(l[0]*(600)))/l[1]))
            end_point =  (1,int((-l[2]-l[0])/l[1]))
            color = (0, 255, 0)
            thickness = 1

            cv2.line(img1copy, start_point, end_point, color, thickness)
            cv2.circle(img1copy,(int(p1i[0]),int(p1i[1])),radius=3,color=(0,0,255),thickness=-1)    
            # cv2.imshow('epi1',img1copy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        for i in range(N):   
            # % The product l=E*p2 is the equation of the epipolar line corresponding
            # % to p2, in the first image.  Here, l=(a,b,c), and the equation of the
            # % line is ax + by + c = 0.
            p2i = np.concatenate((np.array(p2)[:,i],[1]))
            p1i = np.concatenate((np.array(p1)[:,i],[1]))
            l = np.dot(F, p1i.T)
            # print("p2 = ",p2i,"\n")
            # % Calculate residual error.  The product p1'*E*p2 should = 0.  The
            # % difference is the residual.
            res = np.dot(np.dot(p2i.T, F) , p1i)
            if(res > 0.1):
                print("res = ",res,"\n")
            start_point = (600,int((-l[2]-(l[0]*(600)))/l[1]))
            end_point =  (1,int((-l[2]-l[0])/l[1]))
            color = (0, 255, 0)
            thickness = 1

            cv2.line(img2copy, start_point, end_point, color, thickness)
            cv2.circle(img2copy,(int(p2i[0]),int(p2i[1])),radius=3,color=(0,0,255),thickness=-1)
            # cv2.imshow('epi2',img2copy)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        fig =  plt.figure()   
        fig.add_subplot(1,2,1) 
        plt.imshow(img1copy)
        fig.add_subplot(1,2,2) 
        plt.imshow(img2copy)
        plt.show() 
        points, R1, t1, mask = cv2.recoverPose(E, p1.T,p2.T, K)
        print(np.hstack((R1,t1)))
        P2_t = np.hstack((R1,t1))
        P2_temp = np.dot(K,np.hstack((R1,t1)))
        
        
        P2 = np.hstack((R1.T,np.dot(-R1,t1)))
        # rvec,_  =cv2.Rodrigues(R1)
        # print("rotation",R1)
        # print(t1)
        # Hresult_c2_c1 = compute_pose_from_essential(E)  
        # # print("spltpe",Hresult_c2_c1.shape) 
        M1 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])
        M1_temp = np.dot(K,M1)
        points_4d_hom = cv2.triangulatePoints(M1_temp,P2_temp,p1[0:2],p2[0:2])
        points_3D = cv2.convertPointsFromHomogeneous(points_4d_hom.transpose())
        # for i ,P2 in enumerate(Hresult_c2_c1):
        #     A = np.concatenate((np.matmul(p1x, M1) ,np.matmul(p2x , P2)))
        # #     % The solution to AP=0 is the singular vector of A corresponding to the
        # #     % smallest singular value; that is, the last column of V in A=UDV'
        #     [U,D,V] = np.linalg.svd(A,full_matrices=False)
        #     P = np.ravel(V[-1, :4])
        #     d1 = P/P[3]
        #     P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        #     d2 = np.dot(P2_homogenous[:3, :4], d1)

        #     if d1[2] > 0 and d2[2] > 0:
        #         ind = i
        # P2 = np.linalg.inv(np.vstack([Hresult_c2_c1[ind], [0, 0, 0, 1]]))[:3, :4]   
        # print("P2_temp  ",P2_temp)
        # cv2.triangulatePoints()
        # tripoints3d = linear_triangulation(p1, p2, np.dot(K,M1), np.dot(K,P2))

        # tripoints3d = tripoints3d/tripoints3d[3]
def new_2points(lst1,lst2):
    lst_not = [lst2.index(val) for val in lst2 if val not in lst1]
    return lst_not        
def intersection(lst1,lst2):
    lst = [(lst2.index(val) , lst1.index(val)) for val in lst2 if val in lst1]
    
    return lst
def test_reproj_pnp_points(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K, rep_thresh=5):
    """
    Reprojects points fed into Pnp back onto camera whose R and t were just obtained via Pnp.
    Used to assess how good the resection was.
    :param pts3d_for_pnp: List of axis aligned 3D points
    :param pts2d_for_pnp: List of axis aligned 2D points
    :param R_new: Rotation matrix of newly resected image
    :param t_new: Translation vector of newly resected image
    :param rep_thresh: Number of pixels reprojected points must be within to qualify as inliers
    """
    errors = []
    projpts = []
    inliers = []
    # print("len",len(pts3d_for_pnp))
    # print(pts2d_for_pnp[0])
    for i in range(len(pts3d_for_pnp)):
        Xw = pts3d_for_pnp[i]
        # pri?\nt(Xw)
        Xr = np.dot(R_new, Xw.T).reshape(3,1)
        Xc = Xr + t_new
        x = np.dot(K, Xc)
        # print(x)
        x /= x[2]
        # print(x[0])
        # print(pts2d_for_pnp[i][0])
        errors.append([np.float64(x[0] - pts2d_for_pnp[i][0]), np.float64(x[1] - pts2d_for_pnp[i][1])])
        projpts.append(x)
        if abs(errors[-1][0]) > rep_thresh or abs(errors[-1][1]) > rep_thresh: inliers.append(0)
        else: inliers.append(1)
    a = 0
    for e in errors:
        a = a + abs(e[0]) + abs(e[1])
    avg_err = a/(2*len(errors))
    perc_inliers = sum(inliers)/len(inliers)

    return errors, projpts, avg_err, perc_inliers    


def do_pnp(pts3d_for_pnp, pts2d_for_pnp, K, iterations=200, reprojThresh=5):
    """
    Performs Pnp with Ransac implemented manually. The camera pose which has the most inliers (points which
    when reprojected are sufficiently close to their keypoint coordinate) is deemed best and is returned.
    :param pts3d_for_pnp: list of index aligned 3D coordinates
    :param pts2d_for_pnp: list of index aligned 2D coordinates
    :param K: Intrinsics matrix
    :param iterations: Number of Ransac iterations
    :param reprojThresh: Max reprojection error for point to be considered an inlier
    """
    list_pts3d_for_pnp = pts3d_for_pnp
    list_pts2d_for_pnp = pts2d_for_pnp
    pts3d_for_pnp = np.array(pts3d_for_pnp)
    # pts2d_for_pnp = np.expand_dims(np.squeeze(np.array(pts2d_for_pnp)), axis=1)
    # print(pts3d_for_pnp)
    # print(pts2d_for_pnp.shape)
    num_pts = len(pts3d_for_pnp)
    print(num_pts)
    highest_inliers = 0
    for j in range(iterations):
        pt_idxs = np.random.choice(num_pts, 6, replace=False)
        pts3 = np.array([pts3d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        # print("pts",pts3)
        pts2 = np.array([pts2d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        _, rvec, tvec = cv2.solvePnP(pts3, pts2, K, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        pnp_errors, projpts, avg_err, perc_inliers = test_reproj_pnp_points(list_pts3d_for_pnp, list_pts2d_for_pnp, R, tvec, K, rep_thresh=reprojThresh)
        if highest_inliers < perc_inliers:
            highest_inliers = perc_inliers
            best_R = R
            best_tvec = tvec
    R = best_R
    tvec = best_tvec
    # print('rvec:', rvec,'\n\ntvec:', tvec)
    print("avg",avg_err)
    print("inlier",perc_inliers)
    return R, tvec
rvec_l,_ = cv2.Rodrigues(np.eye(3,3))
points_3D = np.squeeze(points_3D)
# print(points_3D)
print(rvec_l)
tvec_l = np.zeros((3,1))    
projPoints_l,_ = cv2.projectPoints(points_3D, rvec_l, tvec_l, K, distCoeffs=np.array([]))    
rvec_r,_ = cv2.Rodrigues(P2_t[0:3,0:3])
t1 = P2_t[:,3]
projPoints_r,_ = cv2.projectPoints(points_3D,rvec_r,t1,K,distCoeffs=np.array([]))
projPoints_l = np.squeeze(projPoints_l)
projPoints_r = np.squeeze(projPoints_r)
# fig = plt.figure()
# fig.suptitle('3D reconstructed', fontsize=16)
# ax = fig.add_subplot(1,1,1,projection='3d')
# ax.plot(points_3D[:,0],points_3D[:,1],points_3D[:,2], 'b.')
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# ax.set_zlabel('z axis')
# ax.title.set_text('PnP_RANSAC')
# # print(len(common_3d))
# ax.view_init(elev=135, azim=90)
# # ax.title()
# plt.show()   
# print(projPoints_l)
# print(projPoints_l.shape)
delta_l = []
delta_r = []
for i in range(len(projPoints_l)):
    delta_l.append(abs(projPoints_l[i][0] - p1[0][i]))
    delta_l.append(abs(projPoints_l[i][1] - p1[1][i]))
    delta_r.append(abs(projPoints_r[i][0] - p2[0][i]))
    delta_r.append(abs(projPoints_r[i][1] - p2[1][i]))
avg_error_l = sum(delta_l)/len(delta_l)   
avg_error_r = sum(delta_r)/len(delta_r) 
print(avg_error_l)
print(avg_error_r)
p_k2 = p2
p3d = np.empty((0,3),float)
# p3d = points_3D 
POINTS_3D = set()
# for i in range(points_3D.shape[0]):
#     POINTS_3D.add(str(points_3D[i]))
sum1 = 0
# print("sum",sum1)
k = 2
dic={}
poses = []
rvecs = [[0],[0],[0]]
tvecs = [[0],[0],[0]] 
poses.append(np.vstack((rvecs,tvecs)))
poses.append(np.vstack((rvec_r,np.reshape(np.array(t1),(3,1)))))
n_cameras = 7
p_k3 = p1
with open('data.txt','w') as f:
    
    # for i in range(len(p1.T)):
    #     f.write("%s " % str(0))
    #     f.write("%s "% str(i))
    #     f.write("%s "% p1[:,i][0])
    #     f.write("%s\n"% p1[:,i][1])
    #     f.write("%s "% str(1))
    #     f.write("%s "%str(i))
    #     f.write("%s "% p2[:,i][0])
    #     f.write("%s\n"% p2[:,i][1])
    for k in range(3,n_cameras+1):
        img2 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/images/dino/img{}.ppm'.format(k-1))
        img3 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/images/dino/img{}.ppm'.format(k))
        # img3 = cv2.resize(img3, (0,0), fx=0.2, fy=.15)
        # img2 = cv2.resize(img2, (0,0), fx=0.2, fy=.15)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_3, descriptors_3 = sift.detectAndCompute(img3,None)
        p_k1,p_k = find_corresponding(img2,img3)

        p_k2 = p_k2.T
        p_k1 = p_k1.T
        p_k = p_k.T
        tup = intersection(p_k2.tolist(),p_k1.tolist())
        indices1 = []
        indices2 = []
        for i,item in enumerate(tup):
            indices2.append(tup[i][0])
            indices1.append(tup[i][1])
        indices1 =  list(set(indices1))
        indices2 =  list(set(indices2))    
        common_3d = [points_3D[i] for i in indices1]
        # common_3d = [tripoints3d.T[i,0:3] for i in indices1]
        # print(common_3d)
        
        impoints = [p_k[i] for i in indices2]
        # for i in range(len(impoints)):   
        #     tup = str(impoints[i][0])+" "+str(impoints[i][1])
        #     if tup not in dic:
        #         dic[tup] = indices1[i]
        #     f.write("%s "%(k-1))
        #     f.write("%s "%dic[tup])
        #     f.write("%s "%impoints[i][0])
        #     f.write("%s\n"%impoints[i][1])
        
        # plt.ion()
        # # fig = plt.figure()
        # fig = plt.figure()
        # fig.suptitle('3D reconstructed', fontsize=16)
        # ax = fig.add_subplot(1,1,1,projection='3d')
        # ax.set_xlabel('x axis')
        # ax.set_ylabel('y axis')
        # ax.set_zlabel('z axis')
        # plt.show()
        # _,r,tvecs,inliers = cv2.solvePnPRansac(np.array(common_3d),np.array(impoints),K,cv2.SOLVEPNP_ITERATIVE)
        R,tvecs = do_pnp(common_3d,impoints,K)
        # R,Jac = cv2.Rodrigues(r)
        Pose3 = np.hstack((R,tvecs))
        r,Jac = cv2.Rodrigues(R)
        # print("RRRR",r)
        poses.append(np.vstack((r,tvecs)))
        # poses.append(r)
        # print(np.dot(K,Pose3))
        N =  np.array(p_k).shape[0]


        new_ind2 = [ind for ind in range(N) if ind not in indices2]
        # print(new_ind2)
        p3 = [p_k[ind] for ind in new_ind2]
        p2 = [p_k1[ind] for ind in new_ind2]
        # print(p2)
        p3 = np.array(p3).T
        p2 = np.array(p2).T
        print("len",len(p3.T))
        # for i in range(len(p3.T)):
        #     sum1+=1
        #     tup = str(p3[:,i][0])+" "+str(p3[:,i][1])
        #     dic[tup] = sum1
        #     f.write("%s "%(k-2))
        #     f.write("%s "%sum1)
        #     f.write("%s "%p2[:,i][0])
        #     f.write("%s\n"%p2[:,i][1])
        #     f.write("%s "%(k-1))
        #     f.write("%s "%sum1)
        #     f.write("%s "%p3[:,i][0])
        #     f.write("%s\n"%p3[:,i][1])
        impoints = np.array(impoints)
        # tup1 = intersection(p3.tolist(),impoints.tolist())
        # print(tup1)
        N = p2.shape[1]
        # p3n = np.array(np.dot(np.linalg.inv(K),np.vstack((p3,np.ones((1,N))))))
        # p2n = np.array(np.dot(np.linalg.inv(K),np.vstack((p2,np.ones((1,N))))))
        # print(p2)
        # print(p3)
        print(Pose3)
        # img3copy = img3.copy()
        # for i in range(N):
        #     cv2.circle(img3copy,(int(p3[0,i]),int(p3[1,i])),3,(255,0,0))
        # cv2.imshow('img',img3copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img2copy = img2.copy()
        # for i in range(N):
        #     cv2.circle(img2copy,(int(p2[0,i]),int(p2[1,i])),3,(255,0,0))
        # cv2.imshow('img',img2copy)
        # cv2.waitKey(0)Pose3 = np.hstack((R,tvecs))
        # cv2.destroyAllWindows()  
        
        tripoints3dnew = cv2.triangulatePoints(P2_temp,np.dot(K,Pose3),p2,p3)
        tripoints3dnew = cv2.convertPointsFromHomogeneous(tripoints3dnew.transpose())
        tripoints3dnew = np.squeeze(tripoints3dnew)
        # print(tripoints3dnew[:,0])
        tripoints3dnew = linear_triangulation(p2, p3, P2_temp,np.dot(K,Pose3))[0:3].T
        projPoints,_ = cv2.projectPoints(np.vstack((np.array(common_3d),tripoints3dnew)), R, tvecs, K, distCoeffs=np.array([])) 
        P2_temp = np.dot(np.linalg.inv(K),P2_temp)
        projPoints_l,_ = cv2.projectPoints(np.vstack((np.array(common_3d),tripoints3dnew)), P2_temp[0:3,0:3], P2_temp[:,3], K, distCoeffs=np.array([]))  
    
        projPoints = np.squeeze(projPoints)
        thresh = 1
        inlier_3D=[]
        print("P_K0",p_k.shape)
        p_k = np.vstack((impoints,p3.T))
        print("P_K",p_k.shape)
        # print("comm",len(common_3d))
        # flag=0
        for i in range(len(projPoints)):
            if abs(projPoints[i][0] - p_k[i][0]) < thresh and abs(projPoints[i][1] - p_k[i][1]) < thresh:
                # print("blaa blaa")
                if i < len(common_3d):
                    # print("sum1")
                    if(str(common_3d[i]) not in POINTS_3D):
                        POINTS_3D.add(str(common_3d[i]))
                        inlier_3D.append(common_3d[i])
                        tup = str(impoints[i][0])+" "+str(impoints[i][1])
                        if tup not in dic:
                            dic[tup] = sum1
                        f.write("%s "%(k-3))
                        f.write("%s "%dic[tup])
                        f.write("%s "%p_k3[:,indices1[i]][0])
                        f.write("%s\n"%p_k3[:,indices1[i]][1])
                        f.write("%s "%(k-2))
                        f.write("%s "%dic[tup])                            
                        f.write("%s "%p_k2[indices1[i]][0])
                        f.write("%s\n"%p_k2[indices1[i]][1])    
                        f.write("%s "%(k-1))
                        f.write("%s "%dic[tup])
                        f.write("%s "%impoints[i][0])
                        f.write("%s\n"%impoints[i][1])    
                        sum1+=1
                else:
                    # if(flag==0):
                        # print("SUM1 ",sum1)
                    # flag=1    
                    tup = str(p3[:,i-len(common_3d)][0])+" "+str(p3[:,i-len(common_3d)][1])
                    dic[tup] = sum1
                    f.write("%s "%(k-2))
                    f.write("%s "%sum1)
                    f.write("%s "%p2[:,i-len(common_3d)][0])
                    f.write("%s\n"%p2[:,i-len(common_3d)][1])
                    f.write("%s "%(k-1))
                    f.write("%s "%sum1)
                    f.write("%s "%p3[:,i-len(common_3d)][0])
                    f.write("%s\n"%p3[:,i-len(common_3d)][1])
                    inlier_3D.append(tripoints3dnew[i-len(common_3d)])
                    sum1+=1
        # print("SUM1 ",sum1)
        delta_l = []
        delta_r = []
        for i in range(len(projPoints)):
            # delta_l.append(abs(projPoints_l[i][0] - p_k1[i][0]))
            # delta_l.append(abs(projPoints_l[i][1] - p_k1[i][1]))
            delta_r.append(abs(projPoints[i][0] - p_k[i][0]))
            delta_r.append(abs(projPoints[i][1] - p_k[i][1]))
        # avg_error_l = sum(delta_l)/len(delta_l)   
        avg_error_r = sum(delta_r)/len(delta_r) 
        # print("avg_img2 ", avg_error_l)    
        # print("avg_img3",avg_error_r)
        p3d = np.vstack((np.array(p3d),np.array(inlier_3D)))
        # print(p3d)
        # print("COMMON_3D",len(common_3d))
        # print("new_points",len(p3.T))
        fig = plt.figure()
        fig.suptitle('3D reconstructed', fontsize=16)
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.plot(p3d[:,0],p3d[:,1], p3d[:,2], 'b.')
        # ax.plot(np.array(common_3d)[:,0],np.array(common_3d)[:,1], np.array(common_3d)[:,2], 'b.')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.title.set_text('PnP_RANSAC')
        ax.view_init(elev=135, azim=90)
        # ax.title()
        plt.show()        
        # print(np.array(impoints).shape)
        # print(p3.shape)
        p_k2 = np.vstack((impoints,p3.T)).T
        print("P_K2 ",len(p_k2.T))
        points_3D = np.vstack((np.array(common_3d),tripoints3dnew))
        P2_temp = np.dot(K,Pose3)
        p_k3 = p_k1.T
        print("P_K3 ",len(p_k3.T))
        # print("p3d",p3d.shape)
print(p3d.shape)    
poses = np.squeeze(np.array(poses))
print(poses.shape)
obs=0
obs1=0
lines_seen = set() # holds lines already seen
outfile = open("final_data.txt", "w")
for line in open("data.txt", "r"):
    obs1+=1
    if line not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
        obs+=1
# print("no_dup_obs",obs1)
for i in range(n_cameras):
    for j in range(6):
        outfile.write(str(poses[i,j])+str("\n"))
outfile.close()

def read_bal_data(n_cameras,file_name,n_observations):
    with open(file_name, "rt") as file:
        # n_points,n_cameras = map(int, file.readline().split())
        # lines =file.read().splitlines()
        # n_observations = int(lines[-1])

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))
        for i in range(n_observations):
            # print(file.readline())
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = np.int32(camera_index)
            point_indices[i] = np.int32(point_index)
            points_2d[i] = [float(x), float(y)]
        # return camera_indices,point_indices, points_2d
        camera_params = np.empty(n_cameras * 6)
        for i in range(n_cameras * 6):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))
        return camera_indices,point_indices,points_2d,camera_params    

from scipy.sparse import lil_matrix
def sparsity_matrix(n_cameras, n_points, camera_indices, point_indices):
    n = n_cameras*6 + n_points*3
    m = 2*camera_indices.size
    A = lil_matrix((m,n),dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2*i,camera_indices*6+s] = 1
        A[2*i+1,camera_indices*6+s] = 1
    for s in range(3):
        A[2*i,n_cameras*6+point_indices*3+s] = 1
        A[2*i+1,n_cameras*6+point_indices*3+s] = 1
    return A 
def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    # print(camera_params.shape)
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = 2360*np.ones(camera_params.shape[0])
    # np.ones()
    # n = np.sum(points_proj**2, axis=1)
    r = 1
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj    
def fun(params,n_cameras,n_points,camera_indices,point_indices,points_3d , points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    # points_3d = points_3d.T
    # points_3d = params[n_cameras * 7:].reshape((n_points, 3))
    # print(point_indices)
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()    
# n_cameras = 8
n_points = p3d.shape[0]
# print(p3d[0])
print("points",n_points)
print("no_obs ",obs)
camera_ind,point_ind,points_2D,camera_params = read_bal_data(n_cameras,"final_data.txt",obs)
A = sparsity_matrix(n_cameras,n_points,camera_ind,point_ind)
x0 = np.hstack((camera_params.ravel(), p3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_ind, point_ind, p3d, points_2D)
from scipy.optimize import least_squares
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_ind, point_ind,p3d, points_2D))
X = res.x[n_cameras*6:]                    
X = np.reshape(X,(n_points,3))
fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot(X[:,0],X[:,1],X[:,2],'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
ax.title.set_text('Bundle_adjust')     
plt.show()
