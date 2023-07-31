package com.example.demo.common;


import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;

//@Configuration
public class cosfig {
//    @Bean
//    public CorsFilter corsFilter() {
//
//        CorsConfiguration corsConfiguration = new CorsConfiguration();
//        /*是否允许请求带有验证信息*/
//        corsConfiguration.setAllowCredentials(true);
//        /*允许访问的客户端域名*/
//        corsConfiguration.addAllowedOrigin("*");
//        /*允许服务端访问的客户端请求头*/
//        corsConfiguration.addAllowedHeader("*");
//        /*允许访问的方法名,GET POST等*/
//        corsConfiguration.addAllowedMethod("*");
//        UrlBasedCorsConfigurationSource source=new UrlBasedCorsConfigurationSource();
//        source.registerCorsConfiguration("/**", corsConfiguration);
//        return new CorsFilter(source);
//
//    }
}
