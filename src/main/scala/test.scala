import scala.collection.mutable.ArrayBuffer

object test {
  def main(args:Array[String] ) = {
    val testString = "Hello world this is me and I am the best"
    val windowSize = 3
    val stride = 1

    // val array = new ArrayBuffer[String]()

//    val result = testString.split(" ").sliding(windowSize + 1, stride).map(l => (l.slice(0, l.length - 1), l.last))
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