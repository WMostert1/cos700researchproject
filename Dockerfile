FROM fabric8/java-alpine-openjdk8-jdk

ADD ./target/COS700ResearchProject-1.0-SNAPSHOT-jar-with-dependencies.jar ./

ADD StatisticalTests/openml_datasets /tmp/openml_datasets

ENTRYPOINT java -jar COS700ResearchProject-1.0-SNAPSHOT-jar-with-dependencies.jar
CMD [ "-ds", "/tmp/openml_datasets" ]
