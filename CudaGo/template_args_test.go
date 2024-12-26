package main

import (
	"testing"
)

func TestTemplateArgs_SetFileName(t *testing.T) {
	type fields struct {
		Package   string
		FileName  string
		Funcs     []*CuFileFunc
		Constants map[string]string
		Variables map[string]string
		PTXCode   string
	}
	type args struct {
		name string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   string
	}{
		{
			name:   "Test with spaces",
			fields: fields{},
			args:   args{"file name with spaces"},
			want:   "file_name_with_spaces",
		},
		{
			name:   "Test with special characters",
			fields: fields{},
			args:   args{"file name with special characters !@#$%^&*()"},
			want:   "file_name_with_special_characters",
		},
		{
			name:   "Test with numbers",
			fields: fields{},
			args:   args{"file name with numbers 1234567890"},
			want:   "file_name_with_numbers_1234567890",
		},
		{
			name:   "Test with mixed",
			fields: fields{},
			args:   args{"file name with mixed 1234567890 !@#$%^&*()"},
			want:   "file_name_with_mixed_1234567890",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &TemplateArgs{
				Package:   tt.fields.Package,
				FileName:  tt.fields.FileName,
				Funcs:     tt.fields.Funcs,
				Constants: tt.fields.Constants,
				Variables: tt.fields.Variables,
				PTXCode:   tt.fields.PTXCode,
			}
			k.SetFileName(tt.args.name)
			if k.FileName != tt.want {
				t.Errorf("SetFileName() = %v, want %v", k.FileName, tt.want)
			}
		})
	}
}

func TestCuFileFunc_SetName(t *testing.T) {
	type fields struct {
		Name     string
		RawName  string
		GoArgs   []Arg
		CArgs    []Arg
		IsKernel bool
	}
	type args struct {
		name string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &CuFileFunc{
				Name:     tt.fields.Name,
				RawName:  tt.fields.RawName,
				GoArgs:   tt.fields.GoArgs,
				CArgs:    tt.fields.CArgs,
				IsKernel: tt.fields.IsKernel,
			}
			k.SetName(tt.args.name)
		})
	}
}

func TestCuFileFunc_SetArgs(t *testing.T) {
	type fields struct {
		Name     string
		RawName  string
		GoArgs   []Arg
		CArgs    []Arg
		IsKernel bool
	}
	type args struct {
		args []string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := &CuFileFunc{
				Name:     tt.fields.Name,
				RawName:  tt.fields.RawName,
				GoArgs:   tt.fields.GoArgs,
				CArgs:    tt.fields.CArgs,
				IsKernel: tt.fields.IsKernel,
			}
			k.SetArgs(tt.args.args)
		})
	}
}
