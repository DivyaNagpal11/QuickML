/* Regression */
$(document).on('click', '#regression', function(event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {
        let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
        ajax_request("POST", "/regression/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['regression_html']);
                $('#regression_algorithm_select').html(response['models_list_html']);

                let models_data = response["saved_regression_models"];
                if (models_data !== 0) {
                    $('#regression_table_data').addClass('showdiv');
                    let target_feature = response['target_feature'];
                    let x = document.getElementById("regression_selected_target_feature");
                    let option = document.createElement("option");
                    option.text = target_feature;
                    option.value = target_feature;
                    x.add(option, x[0]);
                    $('#regression_selected_target_feature').prop('selectedIndex', 0);
                    document.getElementById("regression_selected_target_feature").disabled = true;
                    models_data.forEach(function(each) {
                        let newserialno = document.getElementById("regression_table_body").rows.length;
                        if (newserialno === undefined) {
                            newserialno = 0;
                        }
                        let table = document.getElementById("regression_table_body");
                        let row = table.insertRow(newserialno);
                        let col1 = row.insertCell(0);
                        let col2 = row.insertCell(1);
                        let col3 = row.insertCell(2);
                        let col4 = row.insertCell(3);
                        let col5 = row.insertCell(4);
                        let col6 = row.insertCell(5);
                        let col7 = row.insertCell(6);
                        col1.innerText = newserialno + 1;
                        col2.innerText = each['algorithm'];
                        col3.innerText = each['train_Rsq']
                        col4.innerText = each['train_adjRsq']
                        col5.innerHTML = '<button class="btn" id="run_regression_algorithm">Run</button>';
                        col6.innerHTML = '<button class="btn" id="save_regression_model_for_api">Publish</button>';
                        if (each['api_url'] !== 0) {
                            col7.innerHTML = '<input type="text" value="' + each['api_url'] + '">'
                        }
                        $('#selecteregressiondalgorithm option[value="' + each["algorithm"] + '"]').remove();
                        $('#selecteregressiondalgorithm').prop('selectedIndex', 0);
                    });
                }
                hideloading();
            }
        });
    }
});

function regression_train_size() {
    let test_size = $("#regression_testdatasize").val();
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
            document.getElementById("regression_traindatasize").innerText = train_size_data;
        } else {
            const train_size_data = (1 - test_size);
            document.getElementById("regression_traindatasize").innerText = train_size_data.toFixed(2);
        }
    }
}

$(document).on('click', '#addregrssionalgorithm', function() {
    showloading();
    let selected_algorithm = $("#selecteregressiondalgorithm").val();
    let newserialno = document.getElementById("regression_table_body").rows.length;
    $("#regression_table_data").addClass("showdiv");
    if (selected_algorithm == null) {
        document.getElementById("alertdata").innerHTML = "<p>please select a algorithm</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {
        $('#selecteregressiondalgorithm option[value="' + selected_algorithm + '"]').remove();
        $('#selecteregressiondalgorithm').prop('selectedIndex', 0);
        let table = document.getElementById("regression_table_body");
        let row = table.insertRow(newserialno);
        let col1 = row.insertCell(0);
        let col2 = row.insertCell(1);
        let col3 = row.insertCell(2);
        let col4 = row.insertCell(3);
        let col5 = row.insertCell(4);
        let col6 = row.insertCell(5);
        //let col7 = row.insertCell(6);
        col1.innerText = newserialno + 1;
        col2.innerText = selected_algorithm;
        col5.innerHTML = '<button class="btn" id="run_regression_algorithm">Run</button>';
        col6.innerHTML = '<button class="btn" id="save_regression_model_for_api">Publish</button>';
        hideloading();
    }
});

$(document).on('click', '#run_regression_algorithm', function() {
    showloading();
    let data_options = [];
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("regression_table_body").getElementsByTagName("tr")[row_index];
    let selected_algorithm = row.cells[1].innerText;
    selected_algorithm = $.trim(selected_algorithm);
    let algorithm_info = [];
    algorithm_info.push(row_index, selected_algorithm);
    let selectedtarget = $("#regression_selected_target_feature").val();
    if (selectedtarget == null) {
        document.getElementById("alertdata").innerHTML = "<p>please select the target feature</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {
        let test_size = Number($("#regression_testdatasize").val());
        data_options.push(selectedtarget, test_size);
        let features_value = getValue_checked()
        let feature;
        if (features_value == null){
            feature = "None";
        }
        else {
            feature = features_value
        }

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": $("#selectprojectbtn").val(),
                    "data_options": JSON.stringify(data_options),
                    "algorithm_info": JSON.stringify(algorithm_info),
                    "features": JSON.stringify(feature)};

        ajax_request("POST", "/regression_algorithms/", data).then(function(data){
            if (data !== null || data !== undefined){
                if(data['data_result'] === 'Error'){
                    document.getElementById("alertdata").innerHTML = "<p>Catagorical Data found in the Dataset</p>";
                    $("#alert").addClass("showdiv");
                    hideloading();
                }else{
                    let algorithms_data = data['data_result'];
                    let rowid = algorithms_data[0];
                    let algorithms_accuracy = algorithms_data[2];
                    let row = document.getElementById("regression_table_body").getElementsByTagName("tr")[rowid];
                    row.cells[2].innerText = algorithms_accuracy[0];
                    row.cells[3].innerText = algorithms_accuracy[1];
                    let summary_data = data['summary'];
                    let newserialno = document.getElementById("regression_table_score_body").rows.length;
                    let table = document.getElementById("regression_table_score_body");
                    let row_summary = table.insertRow(newserialno);
                    let col1 = row_summary.insertCell(0);
                    let col2 = row_summary.insertCell(1);
                    let col3 = row_summary.insertCell(2);
                    let col4 = row_summary.insertCell(3);
                    let col5 = row_summary.insertCell(4);
                    let col6 = row_summary.insertCell(5);
                    col1.innerText = summary_data[0];
                    col2.innerText = summary_data[1];
                    col3.innerText = summary_data[2];
                    col4.innerText = summary_data[3];
                    col5.innerText = summary_data[4];
                    let total_cols = $("#totalcols").val();
                    let unchecked_features = data['unchecked_features']
                    if(unchecked_features.length == null){
                        col6.innerText = "None";
                    }
                    else{
                        col6.innerText = unchecked_features
                    }
                    $("#regOLSSummary").removeClass("hidediv");
                    $('#ols_table').html(data['ols_html']);
                    right_side_bar();
                    hideloading();
                }
            }
        });
    }
});

function getValue_checked()
{
    let checked = [];
     $('#valCheck li').each(function()
     {
         if($(this).find("input.checking").is(":checked"))
         {
             let val = $(this).find("input.checking").val()
             checked.push(val);
         }
     });
     return checked
}

$(document).on('click', '#save_regression_model_for_api', function(event) {
    showloading();
    event.preventDefault();

    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("regression_table_body").getElementsByTagName("tr")[row_index];
    let selected_algorithm = row.cells[1].innerText;
    selected_algorithm = $.trim(selected_algorithm);
    colval = row.cells.length;

    let unvalue = "";
    if($("#regOLSSummary").hasClass("hidediv")){
        unvalue = ""
    }else{
        let newserialno1 = document.getElementById("regression_table_score_body").rows.length;
        let row1 = document.getElementById("regression_table_score_body").getElementsByTagName("tr")[newserialno1 - 1]
        unvalue = row1.cells[5].innerText;
    }

    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);

    if (row.cells[2].innerHTML === "") {
        document.getElementById("alertdata").innerHTML = "<p>please run the algorithm first</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    }else {
        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": project_id,
                    "algorithm": selected_algorithm,
                    "dropFeatures": unvalue};

        ajax_request("POST", "/regression_api_save/", data).then(function(response){
            if (response !== null || response !== undefined){
                let api_url = response["api_url"];
                if(colval == 7)
                {
                    row.deleteCell(6);
                }
                let col = row.insertCell(6);
                col.innerHTML = '<input type="text" value="' + api_url + '">';
                right_side_bar();
                hideloading();
            }
        });
    }
});

$(document).on('click', '#regression_validation', function(event) {
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

        ajax_request("POST", "/regression_validation/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['validation_html']);
                hideloading();
            }
        });
    }
});

function regression_validate_get_info() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let selected_model = $("#validation_regression_selected_algorithm").val();

    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "project_id": project_id,
                "algorithm": selected_model};

    ajax_request("POST", "/regression_validation_data/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#validation_regression_data').html(response['validation_data_html']);
            hideloading();
        }
    });
}

function regression_predict_output() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let algorithm = $("#validation_regression_selected_algorithm").val();
    let input_data = regression_get_input_model_data();
    if(input_data === 0) {
        document.getElementById("alertdata").innerHTML = "<p>Please enter the Values</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    }else {
        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": project_id,
                    "algorithm": algorithm,
                    "input_data": JSON.stringify(input_data)};

        ajax_request("POST", "/regression_prediction/", data).then(function(response){
            if (response !== null || response !== undefined){
                if(response['predicted_result'] == 'Error')
                {
                    document.getElementById("alertdata").innerHTML = "<p>Feature count mismatch</p>";
                    $("#alert").addClass("showdiv");
                    hideloading();
                }else{
                    document.getElementById("regression_prediction_result").innerText = response['predicted_result'];
                    hideloading();
                }
            }
        });
    }
}

function regression_get_input_model_data() {
    let data = [];
    let flag = false;
    $('#regression_validation_input_table tbody tr').each(function() {
        let input_feature = $(this).find('td:eq(0)').text();
        let input_value = $(this).find('td:eq(1) input').val();
        let value = checkValidate(input_value);
        if (value == 0){
            flag = true;
            return;
        } else{
            data.push([$.trim(input_feature), input_value]);
        }
    });
    if(flag){
        return 0;
    }
    else {
        return data
    }
}

function checkValidate(input_value)
{
    if(input_value == '')
    {
        return 0;
    }
    else
    {
        return 1;
    }

}
