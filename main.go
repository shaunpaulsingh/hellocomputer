package main

import (
	"errors"
	"fmt"
	"gocv.io/x/gocv"
	"image"
)

type ImageCV struct {
	mat gocv.Mat
}

func main() {
	deviceID := 2
	xmlFile := "haarcascade_frontalface_default.xml"

	// open webcam
	webcam, err := gocv.VideoCaptureDevice(int(deviceID))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer webcam.Close()

	// open display window
	window := gocv.NewWindow("Hello Computer!")
	defer window.Close()

	// prepare image matrix
	img := gocv.NewMat()
	defer img.Close()

	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load(xmlFile) {
		fmt.Printf("Error reading cascade file: %v\n", xmlFile)
		return
	}

	fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("cannot read device %d\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// detect faces
		rects := classifier.DetectMultiScale(img)
		fmt.Printf("found %d faces\n", len(rects))

		// draw a rectangle around each face on the original image,
		// along with text identifying as "Human"


		// show the image in the window, and wait 1 millisecond



		for _, r := range rects {
			img = img.Region(r)
			break
		}
		window.IMShow(img)

		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

func (icv *ImageCV) Load(inputData []byte) error {
	var err error

	icv.mat, err = gocv.IMDecode(inputData, gocv.IMReadUnchanged)
	if err != nil {
		return errors.New("load image error")
	}

	return nil
}

func (icv *ImageCV) ToBytes() (output []byte, err error) {
	output, err = gocv.IMEncode(gocv.JPEGFileExt, icv.mat)
	if err != nil {
		err = errors.New("image to bytes error")
	}

	return
}

func (icv *ImageCV) Crop(left, top, right, bottom int) *ImageCV {
	croppedMat := icv.mat.Region(image.Rect(left, top, right, bottom))
	resultMat := croppedMat.Clone()
	return &ImageCV{mat: resultMat}
}

func (icv *ImageCV) Resize(width, height int) *ImageCV {
	resizeMat := gocv.NewMat()
	gocv.Resize(icv.mat, &resizeMat, image.Pt(width, height), 0, 0, gocv.InterpolationArea)
	_ = icv.mat.Close()
	icv.mat = resizeMat
	return icv
}

func (icv *ImageCV) FlipTB() *ImageCV {
	dstMat := gocv.NewMatWithSize(icv.mat.Rows(), icv.mat.Cols(), icv.mat.Type())
	gocv.Flip(icv.mat, &dstMat, 0)
	return &ImageCV{mat: dstMat}
}

func (icv *ImageCV) FlipLR() *ImageCV {
	dstMat := gocv.NewMatWithSize(icv.mat.Rows(), icv.mat.Cols(), icv.mat.Type())
	gocv.Flip(icv.mat, &dstMat, 1)
	return &ImageCV{mat: dstMat}
}