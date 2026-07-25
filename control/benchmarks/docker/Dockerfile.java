# polybench-java: warm the Gradle cache (~/.gradle) once with the exact
# dependency set every exercism Java exercise uses (junit-bom 5.10.0 +
# jupiter + assertj 3.25.1). After this, per-exercise `gradle test --offline`
# needs no network for plugins, the wrapper, or test libs.
FROM gradle:8.7-jdk21
USER root
WORKDIR /warm
RUN mkdir -p src/test/java src/main/java \
 && cat > build.gradle <<'EOF'
plugins { id "java" }
repositories { mavenCentral() }
dependencies {
    testImplementation platform("org.junit:junit-bom:5.10.0")
    testImplementation "org.junit.jupiter:junit-jupiter"
    testImplementation "org.assertj:assertj-core:3.25.1"
}
test { useJUnitPlatform() }
EOF
RUN cat > src/test/java/WarmTest.java <<'EOF'
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
class WarmTest { @Test void ok() { assertThat(1).isEqualTo(1); } }
EOF
# Populate the gradle dependency + plugin cache, then drop the warm project.
RUN gradle test --no-daemon --console=plain || true
RUN cd / && rm -rf /warm
WORKDIR /work
