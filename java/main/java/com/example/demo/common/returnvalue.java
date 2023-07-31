package com.example.demo.common;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
public class returnvalue<T> {

    private Integer code; //编码：1成功，0和其它数字为失败

    private String msg; //错误信息

    private T data; //数据

    private Map map = new HashMap(); //动态数据

    public static <T> returnvalue<T> success(T object) {
        returnvalue<T> r = new returnvalue<T>();
        r.data = object;
        r.code=1;
        return r;
    }

    public static <T> returnvalue<T> error(String msg) {
        returnvalue r = new returnvalue();
         r.code=0;
        return r;
    }

    public returnvalue<T> add(String key, Object value) {
        this.map.put(key, value);
        return this;
    }
}
