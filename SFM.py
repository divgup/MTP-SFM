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
img1 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/I1.jpg') 
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
img1 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/I1.jpg')   
img1 = cv2.resize(img1, (0,0), fx=0.2, fy=0.2) 
print(img1.shape)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('/home/divanshu05/7thsem/MTP/3Dreconstruction/I2.jpg')  
img2 = cv2.resize(img2, (0,0), fx=0.2, fy=0.2) 

print(img2.shape)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = drawKeypoints(gray1,keypoints_1)
cv2.imshow('img1',img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

img_2 = drawKeypoints(gray2,keypoints_2)
cv2.imshow('img2',img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.asarray([keypoints_1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([keypoints_2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]
    print(pts1)
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

def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)

def nlTriangulation(p1,p2,F):
    J = np.vstack((np.dot(F.T,p1)[0],np.dot(F.T,p1)[1]))
    J = np.vstack((J,np.vstack((np.dot(F,p2)[0],np.dot(F,p2)[1]))))
    s = (np.dot(F,p2)[0])**2+(np.dot(F,p2)[1])**2 + (np.dot(F.T,p1)[0])**2 + (np.dot(F.T,p1)[1]**2)
    delta = np.dot(p1.T,np.dot(F,p2))*J/s
    x_new = np.vstack((np.vstack((p2[0],p2[1])),np.vstack((p1[0],p1[1])))) - delta
    return x_new

p1,p2 = find_corresponding(img1,img2)
p1 = np.float32(p1)
p2 = np.float32(p2)
N =  p1.shape[1]
print(p1.shape)
# F,_ = cv2.findFundamentalMat(p1.T, p2.T, cv2.FM_RANSAC, 3, 0.99)
K = np.array([[739.07654683  , 0.     ,    223.98406117],
 [  0.     ,    735.13429576 ,404.4608068 ],
 [  0.      ,     0. ,          1.        ]])

#try different K and image captured from camera with that K  
# K = np.array([[1184.6  , 0.     ,    325.28],
#  [  0.     ,    1171 ,255.07 ],
#  [  0.      ,     0. ,          1.        ]])
p1n = np.dot(np.linalg.inv(K),np.vstack((p1,np.ones((1,N)))))
p2n = np.dot(np.linalg.inv(K),np.vstack((p2,np.ones((1,N)))))
E = compute_essential_normalized(p1n, p2n)
img1copy = img1.copy()
img2copy = img2.copy()
p1x = skew(p1n[0])
p2x = skew(p2n[0])
F = np.dot(np.linalg.inv(K).T,np.dot(E,np.linalg.inv(K)))
for i in range(N):   
    # % The product l=E*p2 is the equation of the epipolar line corresponding
    # % to p2, in the first image.  Here, l=(a,b,c), and the equation of the
    # % line is ax + by + c = 0.
    p2i = np.concatenate((np.array(p2)[:,i],[1]))
    p1i = np.concatenate((np.array(p1)[:,i],[1]))
    l = np.dot(F, p2i.T)

    # % Calculate residual error.  The product p1'*E*p2 should = 0.  The
    # % difference is the residual.
    res = np.dot(np.dot(p1i.T, F) , p2i)
    start_point = (600,int((-l[2]-(l[0]*(200)))/l[1]))
    end_point =  (1,int((-l[2]-l[0])/l[1]))
    color = (0, 255, 0)
    thickness = 3
   
    cv2.line(img1copy, start_point, end_point, color, thickness)

for i in range(N):   
    # % The product l=E*p2 is the equation of the epipolar line corresponding
    # % to p2, in the first image.  Here, l=(a,b,c), and the equation of the
    # % line is ax + by + c = 0.
    p2i = np.concatenate((np.array(p2)[:,i],[1]))
    p1i = np.concatenate((np.array(p1)[:,i],[1]))
    l = np.dot(F.T, p1i.T)

    # % Calculate residual error.  The product p1'*E*p2 should = 0.  The
    # % difference is the residual.
    res = np.dot(np.dot(p1i.T, F) , p2i)
    start_point = (600,int((-l[2]-(l[0]*(200)))/l[1]))
    end_point =  (1,int((-l[2]-l[0])/l[1]))
    color = (0, 255, 0)
    thickness = 3
   
    cv2.line(img2copy, start_point, end_point, color, thickness)
fig =  plt.figure()   
fig.add_subplot(1,2,1) 
plt.imshow(img1copy)
fig.add_subplot(1,2,2) 
plt.imshow(img2copy)
plt.show() 
Hresult_c2_c1 = compute_pose_from_essential(E)  
# print("spltpe",Hresult_c2_c1.shape) 
M1 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])
for i ,P2 in enumerate(Hresult_c2_c1):
    A = np.concatenate((np.matmul(p1x, M1) ,np.matmul(p2x , P2)))
#     % The solution to AP=0 is the singular vector of A corresponding to the
#     % smallest singular value; that is, the last column of V in A=UDV'
    [U,D,V] = np.linalg.svd(A,full_matrices=False)
    P = np.ravel(V[-1, :4])
    d1 = P/P[3]
    P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i
P2 = np.linalg.inv(np.vstack([Hresult_c2_c1[ind], [0, 0, 0, 1]]))[:3, :4]   

tripoints3d = linear_triangulation(p1n, p2n, M1, P2)
tripoints3d = tripoints3d/tripoints3d[3]
p1 = cart2hom(p1)
p2 = cart2hom(p2)

def fun(X,P1,P2,p1,p2): #non linear error function 
    f=np.asarray([(p1[0] - (np.dot(P1[0,:],X)/np.dot(P1[2,:],X))) , (p1[1]-(np.dot(P1[1,:],X)/np.dot(P1[2,:],X))) , (p2[0]-(np.dot(P2[0,:],X)/np.dot(P2[2,:],X))) , (p2[1]-(np.dot(P2[1,:],X)/np.dot(P2[2,:],X)))])
    # print(f)
    return f

p1i = p1[:,0]
p2i = p2[:,0]
P1 = np.dot(K,np.hstack((np.identity(3),np.zeros((3,1)))))
M2 = np.dot(K,P2)
x = []
X = np.zeros((4,N))
for i in range(N):
    p1i = p1[:,i]
    p2i = p2[:,i]
    x.append(nlTriangulation(p1i,p2i,F)) 
x = np.array(x).squeeze().T
p1i = cart2hom(x[2:4])
p2i = cart2hom(x[0:2])

#plot 3D reconstruction with and without Sampson Correction
fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.add_subplot(1,2,1,projection='3d')

ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)

X = linear_triangulation(p1i,p2i,P1,M2)
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot(X[0,:], X[1,:], X[2,:], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()


#PnP_RANSAC for calculating correct camera pose from 3D-2D correspondences.
distCoeffs = np.int32([0, 0, 0, 0, 0])
objpoints = np.ascontiguousarray(tripoints3d[:3].T.reshape(N,3))
impoints = np.ascontiguousarray(p2i[:2].T.reshape(N,2))
retval, rvecs, tvecs,inliers	=	cv2.solvePnPRansac(objpoints, impoints, K,cv2.SOLVEPNP_ITERATIVE)
Rt,jac = cv2.Rodrigues(rvecs)