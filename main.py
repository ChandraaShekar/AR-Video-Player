"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MIT License

Copyright (c) 2019 Chandra Shekar Reddy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import cv2
import numpy as np
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher

IS_FOUND = 0

MORPH = 7
CANNY = 250

_width  = 600.0
_height = 420.0
_margin = 0.0

# myVid = []

class myClass:
	video_name = "videos_rep/101.mp4"

opend_video = myClass()

video_capture = cv2.VideoCapture(0)

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

is_recognised = False

i=0
myVid = cv2.VideoCapture(opend_video.video_name)

while True :
	if i == 1 & is_recognised == True : 
		myVid = cv2.VideoCapture(opend_video.video_name)
		print("printed")
		i = 2

	ret, rgb = video_capture.read()

	if ( ret ):

		gray = cv2.cvtColor( rgb, cv2.COLOR_BGR2GRAY )

		gray = cv2.bilateralFilter( gray, 1, 10, 120 )

		edges  = cv2.Canny( gray, 10, CANNY )

		kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )

		closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

		contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
		for cont in contours:
			if cv2.contourArea( cont ) > 5000 :
				
				arc_len = cv2.arcLength( cont, True )
				
				approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )
				
				if ( len( approx ) == 4 ):
					IS_FOUND = 1

					pts_src = np.array( approx, np.float32 )

					h, status = cv2.findHomography( pts_src, pts_dst )
					
					out = cv2.warpPerspective( rgb, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )
					cd = ColorDescriptor((8, 12, 3))
					query = out
					features = cd.describe(query)
					result = Searcher.k_nearest_neighbor(features)
					# searcher = Searcher("index.csv")
					# results = searcher.search(features)
					
					# myVid = cv2.VideoCapture("videos_rep/"+results[0][1].split('.')[0]+".mp4")
					new = result+".mp4"
					if(opend_video.video_name != "videos_rep/"+new):
						print(new)
						opend_video.video_name = "videos_rep/" + result + ".mp4"
						is_recognised = True
						i = 1
					if i == 0 :
						i=1
					rect = cv2.minAreaRect(cont)
					box = cv2.boxPoints(rect)
					box = np.int0(box)

					cv2.drawContours( rgb, [box], -1, ( 0, 0, 0 ), 2 )
					
					epsilon = cv2.arcLength(box,True)
					box_approx = cv2.approxPolyDP(box,0.1 * epsilon, True)

					n_h = box[0][1] - box[2][1]
					n_w = box[0][0] - box[1][0]

					if myVid.isOpened():
						retIm, myImage = myVid.read()
						h1,w1,e1 = myImage.shape
						myImArr = np.array(myImage)

					myImage = cv2.flip(myImage, 0)
					pts1 = np.float32([[0,0],[w1, 0],[0, h1],[w1, h1]])
					pts2 = np.float32([box[1], box[0], box[2], box[3]])
					rows, cols, ch = rgb.shape
					A = cv2.getPerspectiveTransform(pts1,pts2)
					im1Reg = cv2.warpPerspective(myImage, A, (cols,rows))
					mask2 = np.zeros(rgb.shape, dtype=np.uint8)
					roi_corners2 = np.int32(box)

					channel_count2 = rgb.shape[2]
					ignore_mask_color2 = (255,)*channel_count2
					cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

					mask2 = cv2.bitwise_not(mask2)

					masked_image2 = cv2.bitwise_and(rgb, mask2)


					final = cv2.bitwise_or(im1Reg, masked_image2)

					cv2.imshow("Output", final)

				else : pass

		cv2.namedWindow( 'edges')
		cv2.imshow( 'edges', edges )

		cv2.namedWindow( 'Input')
		cv2.imshow( 'Input', rgb )

		if IS_FOUND :
			cv2.namedWindow( 'out')
			cv2.imshow( 'out', out )

		if cv2.waitKey(27) & 0xFF == ord('q') :
			break

		if cv2.waitKey(99) & 0xFF == ord('c') :
			current = str( time.time() )
			cv2.imwrite( 'ocvi_' + current + '_edges.jpg', edges )
			cv2.imwrite( 'ocvi_' + current + '_gray.jpg', gray )
			cv2.imwrite( 'ocvi_' + current + '_org.jpg', rgb )
			print("Pictures saved")

	else :
		print("Stopped")
		break

video_capture.release()
cv2.destroyAllWindows()