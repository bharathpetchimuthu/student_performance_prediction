document.addEventListener(
    "DOMContentLoaded",
    function () {

        const studyHours =
            document.getElementById(
                "study_hours"
            );

        const attendance =
            document.getElementById(
                "attendance"
            );

        const previousMarks =
            document.getElementById(
                "previous_marks"
            );

        const assignmentScore =
            document.getElementById(
                "assignment_score"
            );


        const summaryStudy =
            document.getElementById(
                "summaryStudy"
            );

        const summaryAttendance =
            document.getElementById(
                "summaryAttendance"
            );

        const summaryPrevious =
            document.getElementById(
                "summaryPrevious"
            );

        const summaryAssignment =
            document.getElementById(
                "summaryAssignment"
            );


        function updateSummary() {

            summaryStudy.textContent =
                studyHours.value
                    ? studyHours.value + " hrs"
                    : "—";


            summaryAttendance.textContent =
                attendance.value
                    ? attendance.value + "%"
                    : "—";


            summaryPrevious.textContent =
                previousMarks.value
                    ? previousMarks.value + "/100"
                    : "—";


            summaryAssignment.textContent =
                assignmentScore.value
                    ? assignmentScore.value + "/100"
                    : "—";
        }


        studyHours.addEventListener(
            "input",
            updateSummary
        );

        attendance.addEventListener(
            "input",
            updateSummary
        );

        previousMarks.addEventListener(
            "input",
            updateSummary
        );

        assignmentScore.addEventListener(
            "input",
            updateSummary
        );


        updateSummary();


        // ------------------------------------------------
        // Form validation
        // ------------------------------------------------

        const form =
            document.getElementById(
                "manualForm"
            );

        if (form) {

            form.addEventListener(
                "submit",
                function (event) {

                    const hours =
                        parseFloat(
                            studyHours.value
                        );

                    const attend =
                        parseFloat(
                            attendance.value
                        );

                    const previous =
                        parseFloat(
                            previousMarks.value
                        );

                    const assignment =
                        parseFloat(
                            assignmentScore.value
                        );


                    if (
                        isNaN(hours) ||
                        hours < 0 ||
                        hours > 24
                    ) {

                        alert(
                            "Study hours must be between 0 and 24."
                        );

                        event.preventDefault();

                        return;
                    }


                    if (
                        isNaN(attend) ||
                        attend < 0 ||
                        attend > 100
                    ) {

                        alert(
                            "Attendance must be between 0 and 100."
                        );

                        event.preventDefault();

                        return;
                    }


                    if (
                        isNaN(previous) ||
                        previous < 0 ||
                        previous > 100
                    ) {

                        alert(
                            "Previous marks must be between 0 and 100."
                        );

                        event.preventDefault();

                        return;
                    }


                    if (
                        isNaN(assignment) ||
                        assignment < 0 ||
                        assignment > 100
                    ) {

                        alert(
                            "Assignment score must be between 0 and 100."
                        );

                        event.preventDefault();

                        return;
                    }

                }
            );

        }

    }
);