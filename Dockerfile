FROM fabric8/java-alpine-openjdk8-jdk

COPY ./src/main/resources/data-sets/used /src/main/resources/data-sets/used

ADD ./target/COS700ResearchProject-1.0-SNAPSHOT-jar-with-dependencies.jar ./

RUN mkdir out

ENTRYPOINT java -jar COS700ResearchProject-1.0-SNAPSHOT-jar-with-dependencies.jar
