import com.khan.spark.utils.EncodingData.encoding

import scala.collection.mutable.ArrayBuffer

object test {

  def splitAndTokenize(str: String) = {
    str.split(" ").map(t => {
      println(s"Encoding token: ${t}")
      encoding.encode(t)
    }).map(bpe => bpe.toArray).map(_.mkString(" ")).mkString(" ")
  }

  def main(args:Array[String] ) = {
    val testString = "Hello world, this is me"
    val windowSize = 7
    val stride = 1

    // val array = new ArrayBuffer[String]()

    val output = splitAndTokenize(testString)
    print(output)

//    val result = testString.split(" ").sliding(windowSize + 1, stride).map(l => (l.slice(0, l.length - 1) ++ Array.fill(windowSize
//      - l.length + 1)("0"), l.last))
//
//    result.foreach(r => println(s"Input: ${r._1.mkString(" ")}, Output: ${r._2}"))
//    val it = Iterator(testString)
//
//    while (it.hasNext) {
//      // Store iterator start pos
//      val windowItr = it
//      // Advance windowSize steps
//      val input = windowItr take(windowSize)
//      // Grab output token if possible
//      if (windowItr.hasNext) {
//        val output = windowItr.next()
//        print(s"Input: ${input}, Output: ${output}")
//      }
//      it.take(stride)
//    }

  }
}