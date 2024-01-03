// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit

/// UIView for rendering inference output.
class OverlayView: UIView {

  var dots = [CGPoint]()
  var lines = [Line]()

  override func draw(_ rect: CGRect) {
    for dot in dots {
      drawDot(of: dot)
    }
    for line in lines {
      drawLine(of: line)
    }
  }

  func drawDot(of dot: CGPoint) {
    let dotRect = CGRect(
      x: dot.x - Traits.dot.radius / 2, y: dot.y - Traits.dot.radius / 2,
      width: Traits.dot.radius, height: Traits.dot.radius)
    let dotPath = UIBezierPath(ovalIn: dotRect)

    Traits.dot.color.setFill()
    dotPath.fill()
  }

  func drawLine(of line: Line) {
    let linePath = UIBezierPath()
    linePath.move(to: CGPoint(x: line.from.x, y: line.from.y))
    linePath.addLine(to: CGPoint(x: line.to.x, y: line.to.y))
    linePath.close()

    linePath.lineWidth = Traits.line.width
    Traits.line.color.setStroke()

    linePath.stroke()
  }

  func clear() {
    self.dots = []
    self.lines = []
  }
}

private enum Traits {
  static let dot = (radius: CGFloat(5), color: UIColor.orange)
  static let line = (width: CGFloat(1.0), color: UIColor.orange)
}
