//Classification Start
$(document).on('click', '#classification', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/modelling/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['modelling_html']);
                let models_data = response["saved_models"];
                if (models_data !== 0) {
                    $('#algorithm_table_data').addClass('showdiv');
                    let target_feature = response['target_feature'];
                    let x = document.getElementById("selectedtargetfeature");
                    let option = document.createElement("option");
                    option.text = target_feature;
                    option.value = target_feature;
                    x.add(option, x[0]);
                    $('#selectedtargetfeature').prop('selectedIndex', 0);
                    document.getElementById("selectedtargetfeature").disabled = true;
                    $('#type_of_algorithm').prop('selectedIndex', 1);
                    document.getElementById("type_of_algorithm").disabled = true;
                    models_data.forEach(function (each) {
                        let newserialno = document.getElementById("algorithm_table_body").rows.length;
                        if (newserialno === undefined) {
                            newserialno = 0;
                        }
                        let table = document.getElementById("algorithm_table_body");
                        let row = table.insertRow(newserialno);
                        let col1 = row.insertCell(0);
                        let col2 = row.insertCell(1);
                        let col3 = row.insertCell(2);
                        let col4 = row.insertCell(3);
                        let col5 = row.insertCell(4);
                        let col6 = row.insertCell(5);
                        let col7 = row.insertCell(6);
                        let col8 = row.insertCell(7);
                        let col9 = row.insertCell(8);
                        col1.innerText = newserialno + 1;
                        col2.innerText = each['algorithm'];
                        col3.innerText = each["train_accuracy"];
                        col4.innerText = each["test_accuracy"];
                        col5.innerHTML = each["accuracy"];
                        col6.innerHTML = '<button class="btn" id="edit_input_parameters">Edit</button>';
                        col7.innerHTML = '<button class="btn" id="runalgorithm">Run</button>';
                        col8.innerHTML = '<button class="btn" id="save_model_for_api">Publish</button>';
                        if (each['api_url'] !== 0) {
                            col9.innerHTML = '<input type="text" value="' + each['api_url'] + '">'
                        }
                        $('#selectedalgorithm option[value="' + each["algorithm"] + '"]').remove();
                        $('#selectedalgorithm').prop('selectedIndex', 0);
                    });
                    get_models_list();
                }
                $("#models_add_process").addClass("showdiv");
                hideloading();
            }
        });
    }
});

function train_size() {
    let test_size = $("#testdatasize").val();
    let total_rows = $("#totalrows").val();
    if (test_size >= total_rows) {
        document.getElementById("alertdata").innerHTML = "<p>please enter the test size lesser than total data size</p>";
        $("#alert").addClass("showdiv");
    } else if (test_size <= 0) {
        document.getElementById("alertdata").innerHTML = "<p>please enter proper test size value</p>";
        $("#alert").addClass("showdiv");
    } else {
        if (test_size > 1) {
            const train_size_data = parseInt(total_rows - parseInt(test_size));
            document.getElementById("traindatasize").innerText = train_size_data;
        } else {
            const train_size_data = (1 - test_size);
            document.getElementById("traindatasize").innerText = train_size_data.toFixed(2);
        }
    }
}

function get_models_list() {
    showloading();
    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": $("#selectprojectbtn").val()
    };

    ajax_request("POST", "/get_algorithms/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#models_select').html(response['models_list_html']);
        }
    });
    hideloading();
}

function get_selected_algorithm_parameter() {
    showloading();
    let selected_algorithm = $("#selectedalgorithm").val();

    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "selected_algorithm": selected_algorithm
    };

    ajax_request("POST", "/get_algorithm_parameters/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            let add_html = generate_input_parametrs_html(response);
            document.getElementById('models_input_parameters').innerHTML = "";
            document.getElementById('models_input_parameters').innerHTML = add_html;
            $("#models_input_parameters").addClass("showdiv");
            hideloading();
        }
    });
}

function generate_input_parametrs_html(response) {
    let obj = response['input_parameters'];
    let add_html = '';
    for (let i = 0; i < obj.length; i++) {
        let temp = obj[i];
        add_html += "<div style='width: 25%;display: inline-block'> <p style='display: inline-block;padding-left: 15px; width: 75%;'>" + temp[0] + ": </p>";
        if (temp[1] === "input") {
            add_html += '<input style="width: 25%" type="number" value="' + temp[3] + '"></div>';
        } else {
            add_html += "<select style='width: 25%'>";
            let opt = temp[3];
            for (let j = 0; j < opt.length; j++) {
                add_html += '<option value="' + opt[j] + '">' + opt[j] + "</option>";
            }
            add_html += "</select></div>"
        }
    }
    add_html += '<button class="btn hidediv" style="float:right;" id="save_input_parameters">Save</button>';
    return add_html
}

$(document).on('click', '#addalgorithm', function () {
    showloading();
    let selected_algorithm = $("#selectedalgorithm").val();
    let newserialno = document.getElementById("algorithm_table_body").rows.length;
    $("#algorithm_table_data").addClass("showdiv");
    if (selected_algorithm == null) {
        document.getElementById("alertdata").innerHTML = "<p>please select a algorithm</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {
        $('#selectedalgorithm option[value="' + selected_algorithm + '"]').remove();
        $('#selectedalgorithm').prop('selectedIndex', 0);
        let table = document.getElementById("algorithm_table_body");
        let row = table.insertRow(newserialno);
        let col1 = row.insertCell(0);
        let col2 = row.insertCell(1);
        let col3 = row.insertCell(2);
        let col4 = row.insertCell(3);
        let col5 = row.insertCell(4);
        let col6 = row.insertCell(5);
        let col7 = row.insertCell(6);
        let col8 = row.insertCell(7);
        col1.innerText = newserialno + 1;
        col2.innerText = selected_algorithm;
        col6.innerHTML = '<button class="btn" id="edit_input_parameters">Edit</button>';
        col7.innerHTML = '<button class="btn" id="runalgorithm">Run</button>';
        col8.innerHTML = '<button class="btn" id="save_model_for_api">Publish</button>';
        let input_parameters = [];
        $('#models_input_parameters input, #models_input_parameters select').each(
            function (index) {
                let input = $(this).val();
                input_parameters.push(input)
            }
        );

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": $("#selectprojectbtn").val(),
            "algorithm": selected_algorithm,
            "input_parameters": JSON.stringify(input_parameters)
        };

        ajax_request("POST", "/save_algorithm_input_parameters/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                hideloading();
            }
        });

        $("#models_input_parameters").removeClass("showdiv");
    }
});
$(document).on('click', '#edit_input_parameters', function () {
    showloading();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("algorithm_table_body").getElementsByTagName("tr")[row_index];
    let selected_algorithm = row.cells[1].innerText;
    selected_algorithm = $.trim(selected_algorithm);

    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "selected_algorithm": selected_algorithm,
        "project_id": $('#selectprojectbtn').val()
    };

    ajax_request("POST", "/edit_algorithm_parameters/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            let add_html = generate_input_parametrs_html(response);
            document.getElementById('models_input_parameters').innerHTML = "";
            document.getElementById('models_input_parameters').innerHTML = add_html;
            let x = document.getElementById("selectedalgorithm");
            let option = document.createElement("option");
            option.text = selected_algorithm;
            option.value = selected_algorithm;
            x.add(option, x[0]);
            $('#selectedalgorithm').prop('selectedIndex', 0);
            document.getElementById("selectedalgorithm").disabled = true;
            $("#models_input_parameters").addClass("showdiv");
            $("#save_input_parameters").addClass("showdiv");
            $("#addalgorithm").addClass("hidediv");
            hideloading();
        }
    });
});

$(document).on('click', '#save_input_parameters', function () {
    showloading();
    document.getElementById("selectedalgorithm").disabled = false;
    $("#save_input_parameters").removeClass("showdiv");
    let input_parameters = [];
    $('#models_input_parameters input, #models_input_parameters select').each(
        function () {
            let input = $(this).val();
            input_parameters.push(input)
        }
    );
    let algorithm = $('#selectedalgorithm').val();
    $("#models_input_parameters").removeClass("showdiv");

    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": $("#selectprojectbtn").val(),
        "algorithm": algorithm,
        "input_parameters": JSON.stringify(input_parameters)
    };

    ajax_request("POST", "/save_algorithm_input_parameters/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            hideloading();
        }
    });

    let algorithms_in_table = [];
    $('#algorithm_table_body tr').each(function () {
        algorithms_in_table.push($(this).find('td:eq(1)').text());
    });
    if (algorithms_in_table.includes(algorithm)) {
        $('#selectedalgorithm option[value="' + algorithm + '"]').remove();
        $('#selectedalgorithm').prop('selectedIndex', 0);
    }
    $("#addalgorithm").removeClass("hidediv");
});

$(document).on('click', '#runalgorithm', function () {
    showloading();
    if ($("#save_input_parameters").hasClass("showdiv")) {
        document.getElementById("alertdata").innerHTML = "<p>please save the edited input parameters</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {
        let data_options = [];
        let row_index = $(this).closest('tr').index();
        let row = document.getElementById("algorithm_table_body").getElementsByTagName("tr")[row_index];
        let selected_algorithm = row.cells[1].innerText;
        selected_algorithm = $.trim(selected_algorithm);
        let algorithm_info = [];
        algorithm_info.push(row_index, selected_algorithm);
        let randomstate;
        let scaling;
        let selectedtarget = $("#selectedtargetfeature").val();
        if (selectedtarget == null) {
            document.getElementById("alertdata").innerHTML = "<p>please select the target feature</p>";
            $("#alert").addClass("showdiv");
            hideloading();
        } else {
            let test_size = Number($("#testdatasize").val());
            if ($("#randomstate").is(":checked")) {
                randomstate = 1;
            } else {
                randomstate = 0;
            }
            if ($("#scaling").is(":checked")) {
                scaling = 1;
            } else {
                scaling = 0;
            }
            data_options.push(selectedtarget, test_size, randomstate, scaling);

            let data = {
                'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "project_id": $("#selectprojectbtn").val(),
                "data_options": JSON.stringify(data_options),
                "algorithm_info": JSON.stringify(algorithm_info)
            };

            ajax_request("POST", "/algorithms/", data).then(function (response) {
                if (response !== null || response !== undefined) {
                    let algorithms_data = response['data_result'];
                    let rowid = algorithms_data[0];
                    let algorithms_accuracy = algorithms_data[1];
                    let row = document.getElementById("algorithm_table_body").getElementsByTagName("tr")[rowid];
                    row.cells[2].innerText = algorithms_accuracy[0];
                    row.cells[3].innerText = algorithms_accuracy[1];
                    row.cells[4].innerText = algorithms_accuracy[2];
                    right_side_bar();
                    hideloading();
                }
            });
        }
    }
});

$(document).on('click', '#save_model_for_api', function () {
    showloading();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("algorithm_table_body").getElementsByTagName("tr")[row_index];
    let selected_algorithm = row.cells[1].innerText;
    selected_algorithm = $.trim(selected_algorithm);
    if (row.cells[2].innerHTML === "") {
        document.getElementById("alertdata").innerHTML = "<p>please run the algorithm first</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": $("#selectprojectbtn").val(),
            "algorithm": selected_algorithm
        };

        ajax_request("POST", "/save_model_for_api/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                let api_url = response["api_url"];
                let col = row.insertCell(8);
                col.innerHTML = '<input type="text" value="' + api_url + '">';
                hideloading();
            }
        });
    }
});
$(document).on('click', '#validation', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": project_id};
        ajax_request("POST", "/validation/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['validation_html']);
                hideloading();
            }
        });
    }
});

function validate_get_info() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let selected_model = $("#validation_selected_algorithm").val();

    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "project_id": project_id,
                "algorithm": selected_model };

    ajax_request("POST", "/validations_data/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#validation_data').html(response['validation_data_html']);
            hideloading();
        }
    });
}

function predict_output() {
    showloading();
    let input_data = get_input_model_data();
    let project_id = $("#selectprojectbtn").val();
    let algorithm = $("#validation_selected_algorithm").val();

    if (input_data == 0) {
        document.getElementById("alertdata").innerHTML = "<p>Please enter the Values</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": project_id,
                    "algorithm": algorithm,
                    "input_data": JSON.stringify(input_data)};

        ajax_request("POST", "/prediction/", data).then(function(response){
            if (response !== null || response !== undefined){
                document.getElementById("prediction_result").innerText = response['predicted_result'];
                hideloading();
            }
        });
    }
}

function get_input_model_data() {
    let data = [];
    let flag = false;
    $('#validation_input_table tbody tr').each(function () {
        let input_feature = $(this).find('td:eq(0)').text();
        let input_value = $(this).find('td:eq(1) input').val();
        let value = checkValidate(input_value);
        if (value == 0) {
            flag = true;
            return;
        } else {
            data.push([$.trim(input_feature), input_value]);
        }
    });

    if (flag) {
        return 0;
    } else {
        return data
    }
}

//Classifications Ends