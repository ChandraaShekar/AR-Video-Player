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
i = 0

while True:
	ret, rgb = video_capture.read()
	if ( ret ):
		