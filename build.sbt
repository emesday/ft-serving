import scala.xml.transform.{RewriteRule, RuleTransformer}
import scala.xml.{Comment, Elem, Node => XmlNode, NodeSeq => XmlNodeSeq}

name := "ft-serving"

val versions = new {
  val scalatest = "2.2.6"
}

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % versions.scalatest % "test"
)

organization := "com.github.mskimm"

isSnapshot := version.value.endsWith("-SNAPSHOT")

scalaVersion := "2.11.8"

sources in (Compile, doc) := Seq.empty

scalacOptions := Seq("-feature", "unchecked", "-encoding", "utf8")

homepage := Some(url("https://github.com/mskimm/ft-serving"))

licenses := Seq("The Apache License, Version 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))

description := "a minimum Scala implementation for serving the fastText model"

scmInfo := Some{ val git = "https://github.com/mskimm/ft-serving.git"; ScmInfo(url(git), s"scm:git:$git", Some(s"scm:git$git"))}

developers := List(Developer("mskimm", "Min Seok Kim", "mskim.org@gmail.com", url("https://github.com/mskimm")))

publishTo := {
  val maven = "https://oss.sonatype.org"
  if (isSnapshot.value)
    Some("Sonatype Snapshots" at s"$maven/content/repositories/snapshots")
  else
    Some("Sonatype Staging" at s"$maven/service/local/staging/deploy/maven2")
}

credentials ++= Seq(Path.userHome / ".ivy2" / ".credentials").filter(_.exists()).map(Credentials.apply)

releasePublishArtifactsAction := PgpKeys.publishSigned.value

pomPostProcess := { (node: XmlNode) =>
  new RuleTransformer(new RewriteRule {
    override def transform(node: XmlNode): XmlNodeSeq = node match {
      case e: Elem if e.label == "dependency" && (e \ "scope").map(_.text).exists(Set("provided", "test").contains) =>
        val Seq(organization, artifact, version, scope) =
          Seq("groupId", "artifactId", "version", "scope").map(x => (e \ x).head.text)
        Comment(s"$scope dependency $organization#$artifact;$version has been omitted")
      case _ => node
    }
  }).transform(node).head
}
